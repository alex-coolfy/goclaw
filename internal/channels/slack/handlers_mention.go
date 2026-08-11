package slack

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"time"

	slackapi "github.com/slack-go/slack"
	"github.com/slack-go/slack/slackevents"

	"github.com/nextlevelbuilder/goclaw/internal/channels"
	"github.com/nextlevelbuilder/goclaw/internal/store"
)

func (c *Channel) handleAppMention(ev *slackevents.AppMentionEvent) {
	ctx := context.Background()
	ctx = store.WithTenantID(ctx, c.TenantID())
	if ev.User == c.botUserID || ev.User == "" {
		return
	}

	// If requireMention is false, message handler already processes all channel messages.
	// Return BEFORE storing dedup key so we don't block the message handler.
	if !c.RequireMention() {
		return
	}

	// Dedup: app_mention may arrive alongside a message event.
	// Shares the same key format as handleMessage so whichever arrives first
	// processes the mention; the other is safely dropped.
	dedupKey := ev.Channel + ":" + ev.TimeStamp
	if _, loaded := c.dedup.LoadOrStore(dedupKey, time.Now()); loaded {
		return
	}

	senderID := ev.User
	channelID := ev.Channel
	content := ev.Text

	displayName := c.resolveDisplayName(senderID)

	if !c.checkGroupPolicy(ctx, senderID, channelID) {
		return
	}

	content = c.stripBotMention(content)
	content = strings.TrimSpace(content)

	if content == "" {
		return
	}

	localKey := channelID
	threadTS := ev.ThreadTimeStamp
	if threadTS != "" {
		localKey = fmt.Sprintf("%s:thread:%s", channelID, threadTS)
	}

	slog.Debug("slack app_mention received",
		"sender_id", senderID, "channel_id", channelID,
		"preview", channels.Truncate(content, 50))

	replyThreadTS := threadTS
	if replyThreadTS == "" {
		replyThreadTS = ev.TimeStamp
	}

	placeholderOpts := []slackapi.MsgOption{
		slackapi.MsgOptionText("Thinking...", false),
	}
	if replyThreadTS != "" {
		placeholderOpts = append(placeholderOpts, slackapi.MsgOptionTS(replyThreadTS))
	}

	_, placeholderTS, err := c.api.PostMessage(channelID, placeholderOpts...)
	if err == nil {
		c.placeholders.Store(localKey, placeholderTS)
	}

	annotated := fromHeader(displayName, true) + "\n" + content
	finalContent := annotated
	if c.HistoryLimit() > 0 {
		finalContent = c.GroupHistory().BuildContext(localKey, annotated, c.HistoryLimit())
	}

	metadata := map[string]string{
		"message_id":      ev.TimeStamp,
		"user_id":         senderID,
		"username":        displayName,
		"channel_id":      channelID,
		"is_dm":           "false",
		"local_key":       localKey,
		"placeholder_key": localKey,
	}
	if replyThreadTS != "" {
		metadata["message_thread_id"] = replyThreadTS
	}

	c.HandleMessage(senderID, channelID, finalContent, nil, metadata, "group")

	// Record thread participation
	if replyThreadTS != "" {
		participKey := channelID + ":particip:" + replyThreadTS
		c.threadParticip.Store(participKey, time.Now())
	}

	c.GroupHistory().Clear(localKey)
}

// isBotMentioned checks if the message text contains <@botUserID>.
func (c *Channel) isBotMentioned(text string) bool {
	return strings.Contains(text, "<@"+c.botUserID+">")
}

// fromHeader builds the "[From: ...]" annotation prepended to group messages.
// When the sender explicitly @mentioned the bot, the header says so — the
// mention tag itself is stripped from the delivered text, so this marker is
// the model's only reliable signal that the message was addressed to it
// (otherwise a message like "@bot @colleague help" arrives as "@colleague
// help" and reads as someone else's conversation).
func fromHeader(displayName string, mentioned bool) string {
	if mentioned {
		return fmt.Sprintf("[From: %s — @mentioned you]", displayName)
	}
	return fmt.Sprintf("[From: %s]", displayName)
}

// shouldRespondInGroup reports whether a channel message may trigger the agent
// when require_mention is on: either the event's own text @mentions the bot, or
// the bot recently participated in the thread (within threadTTL).
//
// eventText MUST be the raw event text only — never content that has had the
// thread parent quoted into it (fetchThreadParentContext): the parent's
// mention tags would otherwise make every reply in a thread started with a
// @mention pass the gate, regardless of require_mention/thread_ttl.
func (c *Channel) shouldRespondInGroup(eventText, channelID, threadTS string) bool {
	if c.isBotMentioned(eventText) {
		return true
	}

	// Thread participation cache: auto-reply in threads where bot previously participated
	if threadTS != "" && c.threadTTL > 0 {
		participKey := channelID + ":particip:" + threadTS
		if lastReply, ok := c.threadParticip.Load(participKey); ok {
			if time.Since(lastReply.(time.Time)) < c.threadTTL {
				slog.Debug("slack: auto-reply in participated thread",
					"channel_id", channelID, "thread_ts", threadTS)
				return true
			}
			c.threadParticip.Delete(participKey)
		}
	}

	return false
}

// stripBotMention removes <@botUserID> from message text.
func (c *Channel) stripBotMention(text string) string {
	return strings.ReplaceAll(text, "<@"+c.botUserID+">", "")
}

// --- Policy checks ---

func (c *Channel) checkDMPolicy(ctx context.Context, senderID, channelID string) bool {
	result := c.CheckDMPolicy(ctx, senderID, c.config.DMPolicy)
	switch result {
	case channels.PolicyAllow:
		return true
	case channels.PolicyNeedsPairing:
		c.sendPairingReply(ctx, senderID, channelID)
		return false
	default:
		return false
	}
}

func (c *Channel) checkGroupPolicy(ctx context.Context, senderID, channelID string) bool {
	groupPolicy := c.config.GroupPolicy
	if groupPolicy == "" {
		groupPolicy = "open"
	}

	// Slack "allowlist" checks both sender and channel ID.
	if groupPolicy == "allowlist" {
		if !c.HasAllowList() {
			return false
		}
		return c.IsAllowed(senderID) || c.IsAllowed(channelID)
	}

	result := c.CheckGroupPolicy(ctx, senderID, channelID, groupPolicy)
	switch result {
	case channels.PolicyAllow:
		return true
	case channels.PolicyNeedsPairing:
		groupSenderID := fmt.Sprintf("group:%s", channelID)
		c.sendPairingReply(ctx, groupSenderID, channelID)
		return false
	default:
		return false
	}
}

func (c *Channel) sendPairingReply(ctx context.Context, senderID, channelID string) {
	ps := c.PairingService()
	if ps == nil {
		return
	}

	if !c.CanSendPairingNotif(senderID, pairingDebounceTime) {
		return
	}

	code, err := ps.RequestPairing(ctx, senderID, c.Name(), channelID, "default", nil)
	if err != nil {
		slog.Warn("slack: failed to request pairing code", "error", err)
		return
	}

	// Security: do not expose pairing code in group channels (visible to all members).
	// Instead, direct admin to CLI or web UI where pending codes are listed.
	var msg string
	if strings.HasPrefix(senderID, "group:") {
		msg = fmt.Sprintf("This channel is not authorized to use this bot.\n\n"+
			"An admin can approve via CLI:\n  goclaw pairing approve %s\n\n"+
			"Or approve via the GoClaw web UI (Pairing section).", code)
	} else {
		msg = fmt.Sprintf("GoClaw: access not configured.\n\nYour Slack user ID: %s\n\nPairing code: %s\n\nAsk the bot owner to approve with:\n  goclaw pairing approve %s",
			senderID, code, code)
	}
	if _, _, err := c.api.PostMessage(channelID, slackapi.MsgOptionText(msg, false)); err != nil {
		slog.Warn("slack: failed to send pairing reply",
			"channel_id", channelID, "error", err)
	}
	c.MarkPairingNotifSent(senderID)
}

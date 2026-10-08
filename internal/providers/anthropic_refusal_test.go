package providers

import (
	"context"
	"strings"
	"testing"
)

// Claude Haiku 5.5 can end a turn with stop_reason "refusal" and no server-side
// fallback. The provider must not report that as a plain "stop" with empty text.

func TestAnthropicParseResponse_Refusal(t *testing.T) {
	adapter, _ := NewAnthropicAdapter(ProviderConfig{APIKey: "sk-test"})

	tests := []struct {
		name        string
		json        string
		wantContent string // exact when non-empty, otherwise "must be non-empty notice"
		keepText    bool
	}{
		{
			name:     "empty content gets a notice",
			json:     `{"content":[],"stop_reason":"refusal","usage":{"input_tokens":1,"output_tokens":0}}`,
			keepText: false,
		},
		{
			name:        "partial text is kept",
			json:        `{"content":[{"type":"text","text":"I can't help with"}],"stop_reason":"refusal","usage":{"input_tokens":1,"output_tokens":4}}`,
			wantContent: "I can't help with",
			keepText:    true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := adapter.FromResponse([]byte(tt.json))
			if err != nil {
				t.Fatal(err)
			}
			if resp.FinishReason != "content_filter" {
				t.Errorf("FinishReason = %q, want content_filter", resp.FinishReason)
			}
			if tt.keepText && resp.Content != tt.wantContent {
				t.Errorf("Content = %q, want %q", resp.Content, tt.wantContent)
			}
			if !tt.keepText && strings.TrimSpace(resp.Content) == "" {
				t.Error("empty refusal must surface a non-empty notice")
			}
		})
	}

	t.Run("end_turn is unchanged", func(t *testing.T) {
		resp, err := adapter.FromResponse([]byte(`{"content":[],"stop_reason":"end_turn","usage":{}}`))
		if err != nil {
			t.Fatal(err)
		}
		if resp.FinishReason != "stop" || resp.Content != "" {
			t.Errorf("got finish=%q content=%q, want stop/empty", resp.FinishReason, resp.Content)
		}
	})
}

func TestAnthropicChatStream_Refusal(t *testing.T) {
	events := []string{
		"event: message_start\n", `data: {"message":{"usage":{"input_tokens":10}}}` + "\n\n",
		"event: message_delta\n", `data: {"delta":{"stop_reason":"refusal"},"usage":{"output_tokens":0}}` + "\n\n",
		"event: message_stop\n", "data: {}\n\n",
	}
	server := newAnthropicSSEServer(t, events)
	p := newTestAnthropicProvider(server.URL)

	result, err := p.ChatStream(context.Background(), ChatRequest{
		Model:    "claude-haiku-5-5",
		Messages: []Message{{Role: "user", Content: "hello"}},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.FinishReason != "content_filter" {
		t.Errorf("FinishReason = %q, want content_filter", result.FinishReason)
	}
	if strings.TrimSpace(result.Content) == "" {
		t.Error("empty refusal must surface a non-empty notice")
	}
}

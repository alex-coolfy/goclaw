package providers

import (
	"encoding/json"
	"testing"
)

// Claude Haiku 5+ rejects sampling parameters and thinking.type=enabled with
// HTTP 400. Haiku 4.5 (and older) must keep its current behaviour.

func TestAnthropicSkipsTemperature_HaikuFamily(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"claude-haiku-5-5", true},
		{"claude-haiku-5", true},
		{"claude-haiku-5-5-20260601", true},
		{"claude-haiku-6", true},
		{"claude-haiku-10-1", true},
		{"Claude-Haiku-5-5", true},
		{"claude-haiku-4-5", false},
		{"claude-haiku-4-5-20251001", false},
		{"claude-haiku-4", false},
		{"claude-3-5-haiku-20241022", false},
		{"claude-3-haiku-20240307", false},
		{"claude-haiku-", false},
		{"claude-haiku-x", false},
		// unchanged families
		{"claude-opus-4-5", false},
		{"claude-opus-4-6", true},
		{"claude-sonnet-4-5-20250929", false},
		{"claude-sonnet-4-6", true},
		{"claude-sonnet-5-5", true},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			if got := anthropicSkipsTemperature(tt.model); got != tt.want {
				t.Errorf("anthropicSkipsTemperature(%q) = %v, want %v", tt.model, got, tt.want)
			}
		})
	}
}

func TestAnthropicUsesAdaptiveThinking_HaikuFamily(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"claude-haiku-5-5", true},
		{"claude-haiku-5", true},
		{"claude-haiku-5-5-20260601", true},
		{"claude-haiku-6", true},
		{"claude-haiku-4-5", false},
		{"claude-haiku-4-5-20251001", false},
		{"claude-3-5-haiku-20241022", false},
		{"claude-haiku-x", false},
		// unchanged families
		{"claude-opus-4-6", false},
		{"claude-opus-4-7", true},
		{"claude-sonnet-4-20250514", false},
		{"claude-sonnet-5-5", true},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			if got := anthropicUsesAdaptiveThinking(tt.model); got != tt.want {
				t.Errorf("anthropicUsesAdaptiveThinking(%q) = %v, want %v", tt.model, got, tt.want)
			}
		})
	}
}

func haikuRequestBody(t *testing.T, model string, opts map[string]any) (map[string]any, string) {
	t.Helper()
	adapter, _ := NewAnthropicAdapter(ProviderConfig{APIKey: "sk-test"})
	data, headers, err := adapter.ToRequest(ChatRequest{
		Model:    model,
		Messages: []Message{{Role: "user", Content: "hi"}},
		Options:  opts,
	})
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if err := json.Unmarshal(data, &body); err != nil {
		t.Fatal(err)
	}
	return body, headers.Get("anthropic-beta")
}

func TestAnthropicAdapterToRequest_Haiku55(t *testing.T) {
	t.Run("no thinking: sampling params omitted", func(t *testing.T) {
		body, _ := haikuRequestBody(t, "claude-haiku-5-5", map[string]any{OptTemperature: 0.7})
		for _, k := range []string{"temperature", "top_p", "top_k", "thinking", "output_config"} {
			if _, has := body[k]; has {
				t.Errorf("%s should be omitted, got %v", k, body[k])
			}
		}
	})

	t.Run("thinking: adaptive plus effort, no budget, no beta header", func(t *testing.T) {
		body, beta := haikuRequestBody(t, "claude-haiku-5-5", map[string]any{
			OptThinkingLevel: "medium",
			OptTemperature:   0.7,
		})
		if beta != "" {
			t.Errorf("adaptive thinking should not send anthropic-beta, got %q", beta)
		}
		thinking, ok := body["thinking"].(map[string]any)
		if !ok {
			t.Fatal("expected thinking config")
		}
		if thinking["type"] != "adaptive" {
			t.Errorf("thinking type = %v, want adaptive", thinking["type"])
		}
		if _, has := thinking["budget_tokens"]; has {
			t.Error("adaptive thinking must not set budget_tokens")
		}
		out, _ := body["output_config"].(map[string]any)
		if out["effort"] != "medium" {
			t.Errorf("output_config = %v, want effort=medium", body["output_config"])
		}
		if _, has := body["temperature"]; has {
			t.Error("temperature should be omitted")
		}
	})
}

func TestAnthropicAdapterToRequest_Haiku45Unchanged(t *testing.T) {
	for _, model := range []string{"claude-haiku-4-5", "claude-haiku-4-5-20251001"} {
		t.Run(model, func(t *testing.T) {
			body, _ := haikuRequestBody(t, model, map[string]any{OptTemperature: 0.7})
			if body["temperature"] != 0.7 {
				t.Errorf("temperature = %v, want 0.7", body["temperature"])
			}

			body, beta := haikuRequestBody(t, model, map[string]any{OptThinkingLevel: "high", OptTemperature: 0.7})
			if beta == "" {
				t.Error("manual thinking should send the interleaved-thinking beta header")
			}
			thinking, _ := body["thinking"].(map[string]any)
			if thinking["type"] != "enabled" {
				t.Errorf("thinking type = %v, want enabled", thinking["type"])
			}
			if b, _ := thinking["budget_tokens"].(float64); int(b) != 32000 {
				t.Errorf("budget_tokens = %v, want 32000", thinking["budget_tokens"])
			}
			if _, has := body["output_config"]; has {
				t.Error("manual thinking must not set output_config")
			}
			if _, has := body["temperature"]; has {
				t.Error("temperature must be dropped when thinking is enabled")
			}
		})
	}
}

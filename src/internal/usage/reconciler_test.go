package usage

import (
	"testing"
	"time"
)

func TestReconcile_Agreement(t *testing.T) {
	head := &UsageRecord{
		RequestID:    "req-1",
		Service:      "llm",
		ConsumerPeer: "head-1",
		ProviderPeer: "worker-1",
		MetricName:   "tokens",
		MetricValue:  1000,
		Timestamp:    time.Now().Unix(),
	}

	worker := &UsageRecord{
		RequestID:    "req-1",
		Service:      "llm",
		ConsumerPeer: "head-1",
		ProviderPeer: "worker-1",
		MetricName:   "tokens",
		MetricValue:  1000,
		Timestamp:    time.Now().Unix(),
	}

	resolved, err := ReconcileRecords(head, worker, 10)
	if err != nil {
		t.Fatalf("ReconcileRecords failed: %v", err)
	}

	if resolved.ResolvedValue != 1000 {
		t.Errorf("Expected ResolvedValue=1000, got %d", resolved.ResolvedValue)
	}
	if resolved.Disputed {
		t.Error("Expected Disputed=false for matching records")
	}
}

func TestReconcile_SmallDifference(t *testing.T) {
	head := &UsageRecord{
		RequestID:    "req-1",
		Service:      "llm",
		ConsumerPeer: "head-1",
		ProviderPeer: "worker-1",
		MetricName:   "tokens",
		MetricValue:  1000,
		Timestamp:    time.Now().Unix(),
	}

	worker := &UsageRecord{
		RequestID:    "req-1",
		Service:      "llm",
		ConsumerPeer: "head-1",
		ProviderPeer: "worker-1",
		MetricName:   "tokens",
		MetricValue:  1050, // 5% difference
		Timestamp:    time.Now().Unix(),
	}

	resolved, err := ReconcileRecords(head, worker, 10)
	if err != nil {
		t.Fatalf("ReconcileRecords failed: %v", err)
	}

	// Should average: (1000 + 1050) / 2 = 1025
	if resolved.ResolvedValue != 1025 {
		t.Errorf("Expected ResolvedValue=1025, got %d", resolved.ResolvedValue)
	}
	if resolved.Disputed {
		t.Error("Expected Disputed=false for small difference")
	}
}

func TestReconcile_LargeDifference(t *testing.T) {
	head := &UsageRecord{
		RequestID:    "req-1",
		Service:      "llm",
		ConsumerPeer: "head-1",
		ProviderPeer: "worker-1",
		MetricName:   "tokens",
		MetricValue:  1000,
		Timestamp:    time.Now().Unix(),
	}

	worker := &UsageRecord{
		RequestID:    "req-1",
		Service:      "llm",
		ConsumerPeer: "head-1",
		ProviderPeer: "worker-1",
		MetricName:   "tokens",
		MetricValue:  1200, // 20% difference
		Timestamp:    time.Now().Unix(),
	}

	resolved, err := ReconcileRecords(head, worker, 10)
	if err != nil {
		t.Fatalf("ReconcileRecords failed: %v", err)
	}

	if !resolved.Disputed {
		t.Error("Expected Disputed=true for large difference")
	}
}

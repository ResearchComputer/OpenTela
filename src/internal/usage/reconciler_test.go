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

func TestReconcile_NilHead(t *testing.T) {
	worker := &UsageRecord{
		RequestID:    "req-1",
		Service:      "llm",
		ConsumerPeer: "head-1",
		ProviderPeer: "worker-1",
		MetricName:   "tokens",
		MetricValue:  1000,
		Timestamp:    time.Now().Unix(),
	}

	_, err := ReconcileRecords(nil, worker, 10)
	if err == nil {
		t.Fatal("Expected error for nil head record, got nil")
	}
	if err.Error() != "head record cannot be nil" {
		t.Errorf("Expected 'head record cannot be nil', got '%s'", err.Error())
	}
}

func TestReconcile_NilWorker(t *testing.T) {
	head := &UsageRecord{
		RequestID:    "req-1",
		Service:      "llm",
		ConsumerPeer: "head-1",
		ProviderPeer: "worker-1",
		MetricName:   "tokens",
		MetricValue:  1000,
		Timestamp:    time.Now().Unix(),
	}

	_, err := ReconcileRecords(head, nil, 10)
	if err == nil {
		t.Fatal("Expected error for nil worker record, got nil")
	}
	if err.Error() != "worker record cannot be nil" {
		t.Errorf("Expected 'worker record cannot be nil', got '%s'", err.Error())
	}
}

func TestReconcile_NegativeMetricValueHead(t *testing.T) {
	head := &UsageRecord{
		RequestID:    "req-1",
		Service:      "llm",
		ConsumerPeer: "head-1",
		ProviderPeer: "worker-1",
		MetricName:   "tokens",
		MetricValue:  -100, // negative value
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

	_, err := ReconcileRecords(head, worker, 10)
	if err == nil {
		t.Fatal("Expected error for negative metric value in head record, got nil")
	}
	if err.Error() != "head record has negative metric value" {
		t.Errorf("Expected 'head record has negative metric value', got '%s'", err.Error())
	}
}

func TestReconcile_NegativeMetricValueWorker(t *testing.T) {
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
		MetricValue:  -50, // negative value
		Timestamp:    time.Now().Unix(),
	}

	_, err := ReconcileRecords(head, worker, 10)
	if err == nil {
		t.Fatal("Expected error for negative metric value in worker record, got nil")
	}
	if err.Error() != "worker record has negative metric value" {
		t.Errorf("Expected 'worker record has negative metric value', got '%s'", err.Error())
	}
}

func TestReconcile_MismatchedRequestID(t *testing.T) {
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
		RequestID:    "req-2", // different request ID
		Service:      "llm",
		ConsumerPeer: "head-1",
		ProviderPeer: "worker-1",
		MetricName:   "tokens",
		MetricValue:  1000,
		Timestamp:    time.Now().Unix(),
	}

	_, err := ReconcileRecords(head, worker, 10)
	if err == nil {
		t.Fatal("Expected error for mismatched request IDs, got nil")
	}
	if err.Error() != "request IDs do not match" {
		t.Errorf("Expected 'request IDs do not match', got '%s'", err.Error())
	}
}

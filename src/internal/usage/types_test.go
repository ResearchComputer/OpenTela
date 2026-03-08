package usage

import (
	"testing"
	"time"
)

func TestUsageRecordValidation(t *testing.T) {
	record := UsageRecord{
		RequestID:    "test-req-1",
		Service:      "llm",
		ConsumerPeer: "head-peer-id",
		ProviderPeer: "worker-peer-id",
		MetricName:   "tokens",
		MetricValue:  1000,
		Timestamp:    time.Now().Unix(),
	}

	if record.RequestID == "" {
		t.Error("RequestID should not be empty")
	}
	if record.MetricValue <= 0 {
		t.Error("MetricValue should be positive")
	}
}

func TestAggregatedUsageWindow(t *testing.T) {
	now := time.Now().Unix()
	agg := AggregatedUsage{
		PeerID:      "peer-1",
		Service:     "llm",
		MetricName:  "tokens",
		TotalValue:  5000,
		RecordCount: 5,
		WindowStart: now - 3600,
		WindowEnd:   now,
	}

	if agg.WindowEnd <= agg.WindowStart {
		t.Error("WindowEnd should be after WindowStart")
	}
}

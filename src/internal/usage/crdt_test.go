package usage

import (
	"context"
	"testing"
	"time"
)

func TestPublishAggregate(t *testing.T) {
	t.Skip("CRDT integration requires full P2P node setup - covered by integration tests")

	ctx := context.Background()
	agg := &AggregatedUsage{
		PeerID:      "peer-1",
		Service:     "llm",
		MetricName:  "tokens",
		TotalValue:  5000,
		RecordCount: 5,
		WindowStart: time.Now().Unix() - 3600,
		WindowEnd:   time.Now().Unix(),
	}

	err := PublishAggregate(ctx, agg)
	if err != nil {
		t.Logf("PublishAggregate error (expected in unit test): %v", err)
	}
}

func TestGetAggregateKey(t *testing.T) {
	peerID := "peer-1"
	service := "llm"
	metricName := "tokens"
	windowStart := int64(1234567890)

	key := getAggregateKey(peerID, service, metricName, windowStart)
	expected := "/usage/peer-1/llm/tokens/1234567890"

	if key.String() != expected {
		t.Errorf("Expected key=%s, got %s", expected, key.String())
	}
}

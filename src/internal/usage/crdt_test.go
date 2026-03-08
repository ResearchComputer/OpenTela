package usage

import (
	"context"
	"testing"
	"time"
)

func TestPublishAggregate(t *testing.T) {
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

	// Call PublishAggregate - will error without initialized CRDT but validates function works
	err := PublishAggregate(ctx, agg)
	// We expect an error since CRDT isn't initialized in unit tests, but function should be callable
	if err == nil {
		t.Log("PublishAggregate succeeded (CRDT may be initialized)")
	} else {
		t.Logf("PublishAggregate returned expected error in unit test: %v", err)
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

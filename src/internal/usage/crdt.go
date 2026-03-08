package usage

import (
	"context"
	"encoding/json"
	"fmt"

	ds "github.com/ipfs/go-datastore"
	"opentela/internal/protocol"
)

// PublishAggregate publishes an aggregated usage record to the CRDT network
func PublishAggregate(ctx context.Context, agg *AggregatedUsage) error {
	store, _ := protocol.GetCRDTStore()

	key := getAggregateKey(agg.PeerID, agg.Service, agg.MetricName, agg.WindowStart)

	data, err := json.Marshal(agg)
	if err != nil {
		return fmt.Errorf("marshalling aggregate: %w", err)
	}

	return store.Put(ctx, key, data)
}

// GetPeerAggregate retrieves an aggregate from a peer
func GetPeerAggregate(ctx context.Context, peerID, service, metricName string, windowStart int64) (*AggregatedUsage, error) {
	store, _ := protocol.GetCRDTStore()

	key := getAggregateKey(peerID, service, metricName, windowStart)

	data, err := store.Get(ctx, key)
	if err != nil {
		return nil, err
	}

	var agg AggregatedUsage
	if err := json.Unmarshal(data, &agg); err != nil {
		return nil, err
	}

	return &agg, nil
}

// getAggregateKey builds the CRDT key for an aggregate
func getAggregateKey(peerID, service, metricName string, windowStart int64) ds.Key {
	return ds.NewKey(CRDTNamespaceUsage).
		ChildString(peerID).
		ChildString(service).
		ChildString(metricName).
		ChildString(fmt.Sprintf("%d", windowStart))
}

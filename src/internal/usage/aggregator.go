package usage

import (
    "sync"
    "time"
)

// AggregatorKey uniquely identifies an aggregation bucket
type AggregatorKey struct {
    PeerID     string
    Service    string
    MetricName string
}

// AggregatorBucket holds accumulated usage for a specific key
type AggregatorBucket struct {
    TotalValue  int64
    RecordCount int64
    WindowStart int64
    LastUpdate  int64
}

// Aggregator manages dynamic aggregation buckets
type Aggregator struct {
    config *BillingConfig
    mu     sync.RWMutex
    buckets map[AggregatorKey]*AggregatorBucket
}

// NewAggregator creates a new aggregator
func NewAggregator(config *BillingConfig) *Aggregator {
    return &Aggregator{
        config:  config,
        buckets: make(map[AggregatorKey]*AggregatorBucket),
    }
}

// AddRecord adds a value to the appropriate bucket
func (a *Aggregator) AddRecord(peerID, service, metricName string, value int64) {
    key := AggregatorKey{PeerID: peerID, Service: service, MetricName: metricName}

    a.mu.Lock()
    defer a.mu.Unlock()

    bucket, exists := a.buckets[key]
    if !exists {
        bucket = &AggregatorBucket{
            WindowStart: time.Now().Unix(),
        }
        a.buckets[key] = bucket
    }

    bucket.TotalValue += value
    bucket.RecordCount++
    bucket.LastUpdate = time.Now().Unix()
}

// ShouldFlush checks if the bucket should be flushed based on triggers
func (a *Aggregator) ShouldFlush(peerID, service, metricName string) bool {
    key := AggregatorKey{PeerID: peerID, Service: service, MetricName: metricName}

    a.mu.RLock()
    bucket, exists := a.buckets[key]
    a.mu.RUnlock()

    if !exists {
        return false
    }

    // Check value threshold
    if bucket.TotalValue >= a.config.ValueThreshold {
        return true
    }

    // Check time ceiling
    elapsed := time.Now().Unix() - bucket.WindowStart
    maxElapsed := int64(a.config.MaxIntervalMinutes * 60)
    if elapsed >= maxElapsed && bucket.RecordCount > 0 {
        return true
    }

    return false
}

// BuildAggregate creates an AggregatedUsage from the bucket and resets it
func (a *Aggregator) BuildAggregate(peerID, service, metricName string) *AggregatedUsage {
    key := AggregatorKey{PeerID: peerID, Service: service, MetricName: metricName}

    a.mu.Lock()
    defer a.mu.Unlock()

    bucket, exists := a.buckets[key]
    if !exists {
        return nil
    }

    agg := &AggregatedUsage{
        PeerID:      peerID,
        Service:     service,
        MetricName:  metricName,
        TotalValue:  bucket.TotalValue,
        RecordCount: bucket.RecordCount,
        WindowStart: bucket.WindowStart,
        WindowEnd:   time.Now().Unix(),
    }

    // Reset bucket
    delete(a.buckets, key)

    return agg
}

// GetValue returns the current accumulated value (for testing)
func (a *Aggregator) GetValue(peerID, service, metricName string) int64 {
    key := AggregatorKey{PeerID: peerID, Service: service, MetricName: metricName}

    a.mu.RLock()
    defer a.mu.RUnlock()

    if bucket, exists := a.buckets[key]; exists {
        return bucket.TotalValue
    }
    return 0
}

// SetWindowStart sets the window start time (for testing)
func (a *Aggregator) SetWindowStart(peerID, service, metricName string, ts int64) {
    key := AggregatorKey{PeerID: peerID, Service: service, MetricName: metricName}

    a.mu.Lock()
    defer a.mu.Unlock()

    if _, exists := a.buckets[key]; !exists {
        a.buckets[key] = &AggregatorBucket{}
    }
    a.buckets[key].WindowStart = ts
}

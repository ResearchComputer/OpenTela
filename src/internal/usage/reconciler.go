package usage

import (
	"errors"
	"math"
	"time"
)

// ReconcileRecords compares head and worker usage records and produces a resolved value
// disputeThresholdPct is the percentage difference above which records are flagged
func ReconcileRecords(head, worker *UsageRecord, disputeThresholdPct int) (*ResolvedUsage, error) {
	// Validate inputs
	if head == nil {
		return nil, errors.New("head record cannot be nil")
	}
	if worker == nil {
		return nil, errors.New("worker record cannot be nil")
	}
	if head.MetricValue < 0 {
		return nil, errors.New("head record has negative metric value")
	}
	if worker.MetricValue < 0 {
		return nil, errors.New("worker record has negative metric value")
	}

	if head.RequestID != worker.RequestID {
		return nil, errors.New("request IDs do not match")
	}
	if head.Service != worker.Service {
		return nil, errors.New("service names do not match")
	}
	if head.MetricName != worker.MetricName {
		return nil, errors.New("metric names do not match")
	}

	// Calculate percentage difference
	diff := math.Abs(float64(head.MetricValue - worker.MetricValue))
	avg := float64(head.MetricValue+worker.MetricValue) / 2

	var pctDiff float64
	if avg > 0 {
		pctDiff = (diff / avg) * 100
	}

	disputed := pctDiff > float64(disputeThresholdPct)

	var resolvedValue int64
	if disputed {
		// For disputed records, use the average but flag for review
		resolvedValue = int64(avg)
	} else if head.MetricValue == worker.MetricValue {
		// Exact match
		resolvedValue = head.MetricValue
	} else {
		// Small difference - use average
		resolvedValue = int64(avg)
	}

	return &ResolvedUsage{
		HeadRecord:    head,
		WorkerRecord:  worker,
		ResolvedValue: resolvedValue,
		Disputed:      disputed,
		ResolvedAt:    time.Now().Unix(),
	}, nil
}

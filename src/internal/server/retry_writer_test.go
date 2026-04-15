package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewRetryableResponseWriter(t *testing.T) {
	rec := httptest.NewRecorder()
	rw := newRetryableResponseWriter(rec, false, 1024)

	assert.NotNil(t, rw)
	assert.False(t, rw.headersSent)
	assert.False(t, rw.failed)
	assert.False(t, rw.wroteHeader)
	assert.False(t, rw.streaming)
	assert.Equal(t, 1024, rw.maxBufferSize)

	// Header() returns an isolated map, not the underlying writer's headers
	rw.Header().Set("X-Test", "value")
	assert.Equal(t, "value", rw.Header().Get("X-Test"))
	assert.Empty(t, rec.Header().Get("X-Test"))
}

func TestRetryableResponseWriter_NonStreaming_BuffersWriteHeader(t *testing.T) {
	rec := httptest.NewRecorder()
	rw := newRetryableResponseWriter(rec, false, 1024)

	rw.WriteHeader(http.StatusOK)

	assert.True(t, rw.wroteHeader)
	assert.Equal(t, http.StatusOK, rw.statusCode)
	// Non-streaming: headers must NOT be sent to underlying
	assert.False(t, rw.headersSent)
}

func TestRetryableResponseWriter_Streaming_CommitsOn2xx(t *testing.T) {
	rec := httptest.NewRecorder()
	rw := newRetryableResponseWriter(rec, true, 1024)
	rw.Header().Set("Content-Type", "text/event-stream")

	rw.WriteHeader(http.StatusOK)

	assert.True(t, rw.headersSent)
	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Equal(t, "text/event-stream", rec.Header().Get("Content-Type"))
}

func TestRetryableResponseWriter_Streaming_BuffersOnNon2xx(t *testing.T) {
	rec := httptest.NewRecorder()
	rw := newRetryableResponseWriter(rec, true, 1024)

	rw.WriteHeader(http.StatusBadGateway)

	assert.True(t, rw.wroteHeader)
	assert.False(t, rw.headersSent)
	assert.Equal(t, http.StatusBadGateway, rw.statusCode)
}

func TestRetryableResponseWriter_Streaming_DetectedFromResponseContentType(t *testing.T) {
	// Writer created as non-streaming, but response Content-Type is SSE.
	// Layer-3 safety net should upgrade to streaming and commit on 2xx.
	rec := httptest.NewRecorder()
	rw := newRetryableResponseWriter(rec, false, 1024)
	rw.Header().Set("Content-Type", "text/event-stream")

	rw.WriteHeader(http.StatusOK)

	assert.True(t, rw.streaming, "should have upgraded to streaming via response Content-Type")
	assert.True(t, rw.headersSent)
	assert.Equal(t, http.StatusOK, rec.Code)
}

func TestRetryableResponseWriter_NonStreaming_BuffersWrite(t *testing.T) {
	rec := httptest.NewRecorder()
	rw := newRetryableResponseWriter(rec, false, 1024)

	rw.WriteHeader(http.StatusOK)
	n, err := rw.Write([]byte("hello"))

	assert.NoError(t, err)
	assert.Equal(t, 5, n)
	assert.Equal(t, "hello", rw.body.String())
	// Nothing sent to underlying yet
	assert.Empty(t, rec.Body.String())
}

func TestRetryableResponseWriter_ImplicitWriteHeader200(t *testing.T) {
	rec := httptest.NewRecorder()
	rw := newRetryableResponseWriter(rec, false, 1024)

	// Write without calling WriteHeader first
	n, err := rw.Write([]byte("data"))

	assert.NoError(t, err)
	assert.Equal(t, 4, n)
	assert.True(t, rw.wroteHeader, "Write should trigger implicit WriteHeader")
	assert.Equal(t, http.StatusOK, rw.statusCode)
}

func TestRetryableResponseWriter_Streaming_PassthroughAfterCommit(t *testing.T) {
	rec := httptest.NewRecorder()
	rw := newRetryableResponseWriter(rec, true, 1024)
	rw.Header().Set("Content-Type", "text/event-stream")

	rw.WriteHeader(http.StatusOK)
	n, err := rw.Write([]byte("data: hello\n\n"))

	assert.NoError(t, err)
	assert.Equal(t, 13, n)
	assert.Equal(t, "data: hello\n\n", rec.Body.String())
}

func TestRetryableResponseWriter_BufferCapExceeded_Commits(t *testing.T) {
	rec := httptest.NewRecorder()
	rw := newRetryableResponseWriter(rec, false, 10) // 10 byte cap

	rw.WriteHeader(http.StatusOK)
	// First write fits
	_, err := rw.Write([]byte("12345"))
	assert.NoError(t, err)
	assert.False(t, rw.headersSent)

	// Second write exceeds cap — should commit
	_, err = rw.Write([]byte("678901234"))
	assert.NoError(t, err)
	assert.True(t, rw.headersSent)
	assert.True(t, rw.bufferExceeded)
	// Both writes should be in the underlying response
	assert.Equal(t, "12345678901234", rec.Body.String())
}

func TestRetryableResponseWriter_BufferCapExceeded_NotRetryable(t *testing.T) {
	rec := httptest.NewRecorder()
	rw := newRetryableResponseWriter(rec, false, 5)

	rw.WriteHeader(http.StatusOK)
	_, _ = rw.Write([]byte("exceeds-cap"))
	rw.markFailed()

	assert.False(t, rw.isRetryable(), "should not be retryable after buffer commit")
}

func TestRetryableResponseWriter_Flush_Noop_BeforeCommit(t *testing.T) {
	rec := httptest.NewRecorder()
	rw := newRetryableResponseWriter(rec, false, 1024)

	// Flush before commit should not panic or forward
	rw.Flush()
	assert.False(t, rw.headersSent)
}

func TestRetryableResponseWriter_Flush_DelegatesAfterCommit(t *testing.T) {
	rec := httptest.NewRecorder()
	rw := newRetryableResponseWriter(rec, true, 1024)
	rw.Header().Set("Content-Type", "text/event-stream")

	rw.WriteHeader(http.StatusOK)
	_, _ = rw.Write([]byte("data: test\n\n"))
	rw.Flush() // should not panic; httptest.ResponseRecorder implements Flusher
	assert.True(t, rw.headersSent)
}

func TestRetryableResponseWriter_FlushToClient(t *testing.T) {
	rec := httptest.NewRecorder()
	rw := newRetryableResponseWriter(rec, false, 1024)
	rw.Header().Set("X-Custom", "val")
	rw.Header().Set("Content-Type", "application/json")

	rw.WriteHeader(http.StatusCreated)
	_, _ = rw.Write([]byte(`{"ok":true}`))
	rw.flushToClient()

	assert.Equal(t, http.StatusCreated, rec.Code)
	assert.Equal(t, "val", rec.Header().Get("X-Custom"))
	assert.Equal(t, "application/json", rec.Header().Get("Content-Type"))
	assert.Equal(t, `{"ok":true}`, rec.Body.String())
}

func TestRetryableResponseWriter_FlushToClient_DefaultsTo200(t *testing.T) {
	rec := httptest.NewRecorder()
	rw := newRetryableResponseWriter(rec, false, 1024)

	// No WriteHeader called — flushToClient should default to 200
	_, _ = rw.Write([]byte("body"))
	rw.flushToClient()

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Equal(t, "body", rec.Body.String())
}

func TestRetryableResponseWriter_MarkFailed_IsRetryable(t *testing.T) {
	rec := httptest.NewRecorder()
	rw := newRetryableResponseWriter(rec, false, 1024)

	rw.markFailed()

	assert.True(t, rw.isRetryable())
}

func TestRetryableResponseWriter_NotRetryableAfterCommit(t *testing.T) {
	rec := httptest.NewRecorder()
	rw := newRetryableResponseWriter(rec, true, 1024)
	rw.Header().Set("Content-Type", "text/event-stream")

	rw.WriteHeader(http.StatusOK)
	rw.markFailed() // fails after commit

	assert.False(t, rw.isRetryable())
}

# Public API rate limits

The public REST API applies rate limits per API key using a token bucket.

## Default limits

The free plan allows 60 requests per minute with bursts of up to 20 requests. The business plan allows 600 requests per minute with bursts of up to 100 requests. Enterprise customers can agree custom limits in their contract.

## Response headers

Every response includes the headers `X-RateLimit-Limit`, `X-RateLimit-Remaining` and `X-RateLimit-Reset`. When a client exceeds the limit, the API returns status 429 Too Many Requests with a `Retry-After` header in seconds.

## Client guidance

Clients should retry 429 and 503 responses with exponential backoff and jitter, starting at one second and capping at sixty seconds. Write requests should send an `Idempotency-Key` header so that a retried payment or order is not created twice.

## Webhooks

Webhook deliveries are not counted against the rate limit. Failed webhook deliveries are retried for up to 24 hours.

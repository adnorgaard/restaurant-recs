# Debugging Guide

## Debug Endpoints

### 1. Check Cache Status
```bash
GET /debug/cache/{place_id}
```

Example:
```bash
curl http://localhost:8000/debug/cache/ChIJLztrVJNZwokRTmcQJheGNR4
```

Returns:
- Restaurant existence
- Image count and details
- Categories and AI tags status
- Overall cache status (COMPLETE, MISSING_AI_TAGS, MISSING_CATEGORIES, etc.)

### 2. Trace Request Flow
```bash
POST /debug/trace
Content-Type: application/json

{
  "query": "Employees Only, New York"
}
```

Example:
```bash
curl -X POST http://localhost:8000/debug/trace \
  -H "Content-Type: application/json" \
  -d '{"query": "Employees Only, New York"}'
```

Returns:
- Step-by-step timing
- Cache hits/misses
- API calls made
- Total request time

## Server Logs

The server now logs detailed performance and debug information:

### Performance Logs
- `[PERF] extract_place_id took X.XXXs` - Time to get place_id
- `[PERF] check_complete_cache took X.XXXs` - Time to check cache
- `[PERF] get_cached_images took X.XXXs` - Time to query images
- `[PERF] download_images_for_ai took X.XXXs` - Time to download from GCS
- `[PERF] analyze_restaurant_images took X.XXXs` - Time for AI analysis
- `[PERF] TOTAL REQUEST TIME: X.XXXs` - Total request time

### Debug Logs
- `[DEBUG] Place ID: ...` - Place ID found
- `[DEBUG] Found X cached images` - Cache status
- `[DEBUG] Cached categories: X, Cached AI tags: Y` - Cache details
- `[DEBUG] All AI tags cached, skipping AI analysis` - Cache hit
- `[DEBUG] Missing AI tags, downloading...` - Cache miss

### Error Logs
- `[ERROR] Failed to upload image...` - GCS upload failure
- `[ERROR] Failed to cache images...` - Caching failure
- `[ERROR] Traceback: ...` - Full error traceback

## Common Issues

### Issue: Images Not Being Cached
**Symptoms:**
- Database shows 0 images
- Every request takes full time
- Logs show "No cached images"

**Debug:**
1. Check `/debug/cache/{place_id}` - see if images exist
2. Check server logs for `[ERROR] Failed to upload image` or `[ERROR] Failed to cache images`
3. Verify GCS credentials are configured
4. Check if `cache_images()` is being called

### Issue: Slow Even With Cache
**Symptoms:**
- Cache exists but still slow
- Logs show cache hits but still downloading

**Debug:**
1. Check `[PERF]` logs to see which step is slow
2. Check if images are being downloaded unnecessarily
3. Verify `get_complete_cached_restaurant_data` is working
4. Check if AI analysis is running when it shouldn't

### Issue: AI Tags Not Cached
**Symptoms:**
- Images cached but AI analysis runs every time
- Logs show "Missing AI tags"

**Debug:**
1. Check `/debug/cache/{place_id}` - see `with_ai_tags` count
2. Check server logs for `[ERROR] Failed to store AI tags`
3. Verify `store_ai_tags_for_images()` is being called
4. Check database: `SELECT ai_tags FROM restaurant_images WHERE restaurant_id = X`

## Quick Debug Checklist

1. ✅ Check if restaurant exists: `/debug/cache/{place_id}`
2. ✅ Check cache status: Look for `cache_status: "COMPLETE"`
3. ✅ Check performance: Look at `[PERF]` logs
4. ✅ Check errors: Look for `[ERROR]` logs
5. ✅ Trace full flow: Use `/debug/trace` endpoint

## Expected Performance

### First Request (No Cache)
- Total time: 10-30 seconds
- Breakdown:
  - Place ID lookup: 0.5-2s
  - Image fetching: 3-10s
  - Categorization: 2-5s
  - AI analysis: 5-15s

### Second Request (Full Cache)
- Total time: < 1 second
- Breakdown:
  - Place ID lookup: 0.01-0.1s (database)
  - Cache check: 0.01-0.05s
  - Return: 0.01s

If second request is > 1 second, check:
- Is cache actually complete?
- Are images being downloaded unnecessarily?
- Is AI analysis running when it shouldn't?


# 💡 Ideas & Roadmap

A living document for tracking future features, optimizations, and improvements.

---

## 🗓️ Next Sprint Focus

_Highlight 3-5 items to focus on in the near term._

1. 
2. 
3. 

## 📅 Recently Completed

_Move items here when done to track progress._

- [x] Quality scoring v1.1 - Unified `image_quality_score` (replaced lighting+blur), added `time_of_day` and `indoor_outdoor` metadata tags
- [x] Quality scoring for images (v1.0)
- [x] SerpAPI integration for additional data
- [x] Bulk processing system
- [x] GCS image caching
- [x] Extract HTML template to `templates/index.html`

---

## 🚀 Features

### High Priority
- [ ] **Personalized recommendations** — Use user interaction history (likes/dislikes/ratings) to personalize recommendation rankings
- [ ] **Saved lists** — Allow users to create and manage lists (e.g., "Date Night", "Client Dinners", "Brunch Spots")
- [ ] **Location-aware search** — Filter recommendations by geographic proximity or neighborhood
- [ ] **Cuisine type filtering** — Tag-based filtering UI for cuisine types (Italian, Japanese, Mexican, etc.)

### Medium Priority
- [ ] **Restaurant comparison view** — Side-by-side comparison of 2-3 restaurants
- [ ] **Share functionality** — Generate shareable links for individual restaurants or lists
- [ ] **Social features** — Follow friends and see their liked restaurants
- [ ] **Restaurant hours & availability** — Integrate opening hours and reservation availability
- [ ] **Price tier filtering** — Filter by estimated price range

### Nice to Have
- [ ] **Group recommendations** — Input multiple users' preferences to find restaurants everyone would enjoy
- [ ] **"Discover" mode** — Surface random, high-quality restaurants outside user's usual preferences
- [ ] **Dietary restriction filtering** — Vegetarian, vegan, gluten-free, etc.
- [ ] **Photo submissions** — Allow users to contribute their own photos

---

## ⚡ Performance & Optimization

- [ ] **Batch embedding generation** — Process multiple restaurants in parallel during bulk imports
- [ ] **Lazy image loading** — Implement proper lazy loading for image grids
- [ ] **CDN for GCS images** — Add Cloud CDN in front of GCS bucket for faster image delivery
- [ ] **Redis caching layer** — Cache frequent database queries (popular restaurants, recent searches)
- [ ] **Connection pooling tuning** — Optimize PostgreSQL connection pool settings
- [ ] **Async Places API calls** — Convert blocking requests to async for better concurrency
- [ ] **Image compression** — Auto-optimize images on upload (WebP conversion, size variants)

---

## 🧠 AI & Machine Learning

- [ ] **Fine-tune image categorization** — Improve food/interior/exterior classification accuracy
- [ ] **Ambiance embeddings** — Generate embeddings specifically for vibe/atmosphere (separate from food)
- [ ] **Multi-modal search** — Allow searching by uploading a reference image ("find places that look like this")
- [ ] **Review sentiment integration** — Incorporate Google/Yelp review sentiment into recommendations
- [ ] **Seasonal recommendations** — Adjust recommendations based on season (patios in summer, cozy spots in winter)
- [ ] **Trending detection** — Identify newly popular restaurants in the area
- [ ] **Confidence scoring** — Show users how confident the AI is in its recommendations

---

## 🏗️ Architecture & Infrastructure

- [ ] **Background job queue** — Celery/RQ for async processing (image analysis, embedding generation)
- [ ] **Rate limiting** — Protect API endpoints from abuse
- [ ] **API versioning** — Implement `/api/v1/` prefix for future backwards compatibility
- [ ] **Health check improvements** — Add readiness vs liveness probes, check external dependencies
- [ ] **Structured logging** — JSON logging with correlation IDs for tracing requests
- [ ] **Metrics & monitoring** — Prometheus metrics, Grafana dashboards
- [ ] **Database migrations CI** — Auto-run Alembic migrations in deployment pipeline

---

## 🎨 Frontend & UX

- [ ] **Mobile-responsive redesign** — Optimize UI for mobile devices
- [ ] **Dark mode** — System preference detection + manual toggle
- [ ] **Map view** — Show recommended restaurants on an interactive map
- [ ] **Image carousel** — Swipeable image gallery for restaurant photos
- [ ] **Skeleton loading states** — Improve perceived performance with skeleton screens
- [ ] **Keyboard shortcuts** — Navigate/search without mouse
- [ ] **Onboarding flow** — Guide new users through initial preferences setup

---

## 🔒 Security & Auth

- [ ] **OAuth providers** — Google/Apple sign-in
- [ ] **Email verification** — Verify email addresses on registration
- [ ] **Password reset flow** — Forgot password functionality
- [ ] **Session management** — View/revoke active sessions
- [ ] **API key authentication** — For programmatic access/integrations
- [ ] **Rate limiting per user** — Prevent individual accounts from overwhelming the system

---

## 📊 Analytics & Insights

- [ ] **Admin dashboard** — View system stats, popular restaurants, user activity
- [ ] **A/B testing framework** — Test different recommendation algorithms
- [ ] **Search analytics** — Track what users search for to improve results
- [ ] **Recommendation feedback** — "Was this helpful?" to improve future recommendations
- [ ] **Cache hit rates** — Monitor caching effectiveness

---

## 🧹 Technical Debt & Cleanup

- [ ] **Split large endpoints** — Break up `/test` and `/classify` into smaller functions
- [ ] **Consistent error handling** — Standardize error response format across all endpoints
- [ ] **Test coverage** — Add unit tests for services, integration tests for endpoints
- [ ] **Type hints completion** — Ensure all functions have proper type annotations
- [ ] **Documentation** — OpenAPI descriptions for all endpoints
- [ ] **Environment validation** — Startup checks for required env vars and connections
- [ ] **Improve version tracking logic** — Current versioning only tracks prompt/logic versions, but doesn't account for changes to the smart fetch flow (e.g., fetching from specific categories vs "All"). Consider adding a "fetch_version" or making versioning more granular to detect when re-processing is actually needed.

---

## 🔌 Integrations

- [ ] **Yelp API** — Cross-reference ratings and reviews
- [ ] **OpenTable/Resy** — Direct reservation links
- [ ] **Uber Eats/DoorDash** — Delivery availability
- [ ] **Instagram** — Pull recent tagged photos
- [ ] **Calendar integration** — Add dinner plans to Google Calendar

---

## 📝 Data Quality

- [ ] **Duplicate detection** — Identify and merge duplicate restaurant entries
- [ ] **Stale data refresh** — Re-analyze restaurants periodically to catch updates
- [x] **Image quality filtering** — Automatically exclude blurry/dark/irrelevant images _(Implemented in quality_service.py with GPT-4 Vision scoring)_
- [ ] **Tag normalization** — Consolidate similar tags (e.g., "cozy" and "intimate")
- [ ] **Manual curation tools** — Admin interface to correct misclassified data
- [ ] **Time-of-day filtering** — Filter images by day/night using `time_of_day` metadata

---

## 🌍 Expansion & Scaling

- [ ] **Multi-city support** — Scale to multiple cities with location-aware defaults
- [ ] **Multi-language** — Internationalization for tags and descriptions
- [ ] **White-label API** — Package as a service for other apps to use
- [ ] **Mobile app** — Native iOS/Android apps

---

## 💭 Experimental Ideas

- [ ] **Voice search** — "Find me a romantic Italian restaurant"
- [ ] **AR view** — Point phone camera at a street to see restaurant info overlays
- [ ] **Taste profile quiz** — Generate recommendations from a fun onboarding quiz
- [ ] **Restaurant DNA** — Visual breakdown of what makes each restaurant unique
- [ ] **Time-of-day context** — Different recommendations for lunch vs dinner vs late night
- [ ] **Weather integration** — Suggest indoor/outdoor seating based on weather

---



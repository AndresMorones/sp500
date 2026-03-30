# Phase 1 Complete Analysis Report — All 7 Tickers

**Date**: 2026-03-29 | **Method**: 3-run LLM consensus per ticker | **Stage**: 3 — Structured News Extraction Schema Design

## Overview

Phase 1 established the taxonomy that governs how news articles will be scored and categorized in Stage 3. For each ticker, three independent LLM runs generated candidate category/dimension schemas, then a consensus run reconciled them. The result is a per-ticker schema with:

- **Categories** — mutually exclusive buckets classifying what a news article is about
- **Dimensions** — 1–10 scoring axes that quantify how impactful, surprising, and long-lasting the article is
- **Rationale** — documented evidence for every keep/drop decision

Raw consensus files: `data/output/news_phase1_raw/<TICKER>_consensus.json`

---

## AAPL — Apple Inc.

### Categories (9 total)

| ID | Label | Core Coverage |
|----|-------|---------------|
| earnings_financial_results | Earnings & Financial Results | Quarterly EPS/revenue vs estimates, buybacks, dividends, guidance revisions |
| market_sector_sentiment | Market & Sector Sentiment | Mag-7 baskets, tech sector rotations, macro-driven mentions, broad market roundups |
| analyst_consensus_signals | Analyst & Consensus Signals | Upgrades, downgrades, price targets, conviction-list adds/removes, ratings |
| product_launches_hardware | Product Launches & Hardware | iPhone, iPad, Mac, Watch, AirPods, Vision Pro, chips, OS releases, demand data |
| ai_strategy_innovation | AI Strategy & Innovation | Apple Intelligence, OpenAI/Gemini partnerships, WWDC AI announcements, Car→AI pivot |
| china_market_competition | China Market & Competitive Dynamics | China iPhone sales, Huawei market-share gains, Tim Cook China visits, brand rivalry |
| regulatory_antitrust_legal | Regulatory, Antitrust & Legal | DOJ lawsuit, EU DMA, Epic/Masimo/App Store litigation, Watch import bans, India/Japan probes |
| supply_chain_manufacturing | Supply Chain & Manufacturing | India/Vietnam expansion, Foxconn/Tata operations, production disruptions, supplier labor |
| corporate_strategy_operations | Corporate Strategy & Operations | Pivots, layoffs, acquisitions, union activity, leadership departures, services strategy |

**Why 9 categories?** AAPL's news landscape is uniquely diverse: a hardware product cycle, a deepening regulatory war on three continents, a geographically concentrated revenue risk in China, and a belated entry into the AI race. No single category can hold all of these without collapsing meaning. The China category is separate because it carries both revenue risk and competitive-threat dimensions that don't exist for any other ticker in the same form.

### Dimensions (14 total — consensus 3/3)

| Dimension | Scale Description | Key Discriminating Examples |
|-----------|------------------|-----------------------------|
| materiality | 1=trivial noise → 10=direct large-scale impact on revenue/margins/TAM | Arcade game launch (1–2) vs $110B buyback or DOJ antitrust (9–10) |
| surprise | 1=fully priced in → 10=completely unexpected | Expected iPhone launch (1–2) vs Apple Car cancellation & AI pivot, DOJ timing (9–10) |
| temporal_horizon | 1=days-to-weeks → 10=structural multi-year shift | iOS bugs (1–2) vs DOJ litigation, EU DMA, India manufacturing build-out (8–10) |
| sentiment_strength | 1=neutral/factual → 10=extreme positive or negative tone | Product spec sheet (1) vs China sales plunge, WWDC AI-rally coverage (8–10) |
| information_density | 1=pure opinion → 10=hard data with specific figures | Foldable iPhone speculation (1–2) vs earnings EPS breakdowns, IDC shipment data (9–10) |
| directional_clarity | 1=ambiguous → 10=unambiguous direction | iPhone 16 contradictory demand data (2–3) vs Barclays downgrade, Apple Watch ban, $110B buyback (9–10) |
| scope | 1=AAPL only → 5=sector-wide → 10=macro/market-wide | Patent dispute (1–2) vs EU DMA (affects all gatekeepers, 5–6) vs global recession selloff (9–10) |
| competitive_impact | 1=no competitive read-through → 10=major shift in competitive landscape | Product color launch (1) vs Huawei doubling shipments overtaking AAPL in China, Samsung global lead (8–9) |
| regulatory_risk | 1=no regulatory angle → 10=structural business model threat | Software update (1) vs DOJ antitrust, EU DMA charges, €500M+ fines (9–10) |
| management_signal | 1=no leadership angle → 10=fundamental strategic pivot signal | Routine software update (1) vs Apple Car cancellation + AI redirect, 600+ layoffs, design chief departure (9–10) |
| expected_duration | 1=resolves within a day → 10=weeks/months of follow-up coverage | iOS bug one-day outage (1) vs DOJ antitrust saga, EU DMA compliance arc (9–10) |
| narrative_shift | 1=reinforces existing thesis → 10=fundamentally changes investment narrative | Routine quarterly refresh (1–2) vs Apple Intelligence reframing hardware stagnation → AI upgrade cycle (9–10) |
| repeatedness | 1=nth rehash of known story → 10=first report of new information | 20th iPhone 16 launch article (1) vs Bloomberg Apple Car scoop, DarwinAI acquisition first-report (9–10) |
| financial_result_surprise | 1=no results or in-line → 10=dramatic beat/miss vs guidance | Non-earnings articles (1) vs Q1 FY24 EPS beat + China miss, Q2 steep iPhone decline + record $110B buyback (7–10) |

### Dropped Dimensions

| Dimension | Dropped by | Reason |
|-----------|-----------|--------|
| controversy | 3/3 runs | AAPL's institutional investor base is aligned; regulatory/labor articles are clearly negative not contested; would cluster near 1–3 for vast majority |
| actionability | 2/3 runs | Nearly every AAPL news item is instantly tradeable given extreme liquidity; also redundant with directional_clarity × information_density combination |

---

## AMZN — Amazon.com, Inc.

### Categories (9 total)

| ID | Label | Core Coverage |
|----|-------|---------------|
| aws_cloud_infrastructure | AWS Cloud & Data Center Investment | Global infrastructure, new regions, Graviton/Trainium chips, re:Invent, enterprise cloud deals |
| ai_generative_ai_strategy | AI & Generative AI Products | Bedrock, Amazon Q, Rufus, Anthropic partnership, AI seller/advertiser tools, Alexa AI overhaul |
| ecommerce_retail_operations | E-Commerce & Retail Operations | Prime Day, delivery speed, low-cost China storefront, international expansion, Temu/Shein competition |
| content_entertainment_sports | Content, Entertainment & Sports | Prime Video (Fallout, Rings of Power), NFL Thursday Night Football, NBA, NWSL, live sports |
| regulatory_legal_antitrust | Regulatory, Legal & Antitrust | FTC antitrust lawsuit, EU DSA, iRobot acquisition collapse, tax disputes, CPSC product liability |
| labor_workforce_organization | Labor, Workforce & Organization | Union organizing, strikes, RTO mandate, Twitch/Alexa/Audible layoffs, AWS CEO transition |
| earnings_financial_results | Earnings & Financial Results | Quarterly reports (Q4 2023–Q3 2024), AWS/advertising/retail segment performance, guidance, margins |
| market_sector_sentiment | Market & Sector Sentiment | Nasdaq corrections, recession fears, Magnificent Seven, Fed decisions, $2T milestone, Dow inclusion |
| analyst_consensus_signals | Analyst & Consensus Signals | Price targets, upgrades/downgrades from BofA, UBS, Jefferies, Wedbush, valuation commentary |

**Why 9 categories?** Amazon's business is unique in the corpus for its operational breadth — it simultaneously operates a cloud hyperscaler, an e-commerce marketplace, a content studio, and a logistics company. AWS and AI are kept separate from e-commerce because their market drivers are entirely different. Labor and workforce disputes are separated because Amazon faced more frequent and material workforce events than any other ticker in this dataset.

### Dimensions (13 total — consensus 3/3)

| Dimension | Scale Description | Key Discriminating Examples |
|-----------|------------------|-----------------------------|
| materiality | 1=no financial relevance → 10=direct large-scale impact on segment financials | How-to deal guides (1) vs multi-billion AWS infrastructure commitments, quarterly earnings (9–10) |
| surprise | 1=fully expected → 10=contradicts prevailing market assumptions | Pre-announced Prime Day date (1) vs five-day RTO mandate, AWS CEO departure, iRobot collapse (8–10) |
| temporal_horizon | 1=single-day event → 10=structural multi-year shift | Prime Day deal roundup (1) vs 15-year data center commitments, nuclear energy supply agreements (9–10) |
| sentiment_strength | 1=neutral product guides → 10=extreme positive/negative tone | Device how-to article (1) vs 6-8% share surge after earnings beat, strike/antitrust coverage (9–10) |
| information_density | 1=pure marketing copy → 10=specific revenue, EPS, deal terms | Prime membership puff piece (1) vs earnings with segment figures and exact dollar commitments (9–10) |
| directional_clarity | 1=ambiguous → 10=unambiguously good or bad | Anthropic investment scrutiny (2–3, unclear if threat or validation) vs clear earnings beats, clear fines (9–10) |
| scope | 1=company-specific → 10=macro/market-wide | New fulfillment center opening (1) vs August 2024 recession scare affecting all equities (9–10) |
| competitive_impact | 1=no competitive shift → 10=fundamentally alters competitive dynamics | Routine fulfillment update (1) vs NBA sports-streaming rights shift, low-cost Temu-rival storefront, nuclear energy arms race (7–9) |
| regulatory_risk | 1=no regulatory angle → 10=structural business change or large penalties | Device launches (1) vs FTC antitrust surviving dismissal, EU blocking iRobot, Italy €121M seizure, France €32M fine (8–10) |
| management_signal | 1=no leadership angle → 10=major strategic pivot or C-suite decision | Routine product article (1) vs AWS CEO transition Selipsky→Garman, Jassy's strategic shareholder letter, five-day RTO, Andrew Ng board appointment (8–10) |
| narrative_shift | 1=reinforces existing thesis → 10=changes dominant narrative | AWS growth articles (1–2, thesis-confirming) vs nuclear energy for AI pivot, $2T valuation, Dow inclusion, Alexa subscription model (7–10) |
| repeatedness | 1=nth rehash → 10=first report of new information | Identical Prime Day deal roundups across outlets (1) vs breaking RTO mandate, AWS CEO change, iRobot collapse (9–10) |
| financial_result_surprise | 1=no results or in-line → 10=dramatic beat/miss | Non-earnings articles (1) vs Q4 2023/Q1 2024 strong beats driving surges, Q2 2024 slowing growth causing 8% decline (7–10) |

### Dropped Dimensions

| Dimension | Dropped by | Reason |
|-----------|-----------|--------|
| actionability | 2/3 runs | Composite of materiality + information_density + directional_clarity; articles scoring high on those three are inherently actionable |
| controversy | 2/3 runs | AMZN articles cluster at low controversy; divisive stories (RTO, union) better captured by sentiment_strength × directional_clarity |
| expected_duration | 3/3 runs | Universally judged redundant with temporal_horizon for Amazon's news corpus |

---

## GOOGL — Alphabet Inc.

### Categories (9 total)

| ID | Label | Core Coverage |
|----|-------|---------------|
| earnings_financial_results | Earnings & Financial Results | Quarterly EPS/revenue, dividend announcements, buyback programs, market cap milestones |
| market_sector_sentiment | Market & Sector Sentiment | Nasdaq/S&P movements, macro reactions, sector rotation, tech trend articles |
| analyst_consensus_signals | Analyst Consensus Signals | Upgrades/downgrades, price targets, broker commentary, institutional positioning |
| antitrust_regulatory_legal | Antitrust, Regulatory & Legal | DOJ search monopoly case, EU fines and DMA, UK CMA probes, Play Store rulings, ad tech trials, forced breakup potential |
| ai_product_strategy | AI & Product Strategy | Gemini launches/controversies, AlphaFold, Trillium/Axion chips, AI Search, DeepMind, Google I/O, Pixel |
| cloud_infrastructure_investment | Cloud & Infrastructure Investment | Global data center construction, enterprise cloud deals, nuclear/renewable energy, submarine cables |
| acquisitions_strategic_investments | Acquisitions & Strategic Investments | Wiz ($23B), HubSpot talks, Character.AI talent, Flipkart/Anthropic investments, licensing deals |
| leadership_workforce_corporate | Leadership, Workforce & Corporate | C-suite changes, layoffs, restructuring, Waymo developments, governance actions |
| advertising_revenue_ecosystem | Advertising & Revenue Ecosystem | Ad tech disputes, cookie deprecation, YouTube monetization, publisher licensing, Play Store policies |

**Why 9 categories?** GOOGL's defining characteristic is the simultaneous existence of a dominant regulatory story (DOJ breakup), a legitimate AI competition narrative (Gemini vs OpenAI), and a core advertising business under structural disruption. Separating antitrust from AI from advertising is critical because these three themes have entirely different valuation implications and temporal horizons.

### Dimensions (13 total — consensus 3/3)

| Dimension | Scale Description | Key Discriminating Examples |
|-----------|------------------|-----------------------------|
| materiality | 1=trivial roundup mention → 10=directly impacts core revenue or enterprise value | Passing market mention (1) vs DOJ breakup proposal, Wiz $23B acquisition talks (9–10) |
| surprise | 1=fully expected → 10=dramatically divergent from expectations | Scheduled earnings date (1) vs unexpected DOJ breakup proposal, Gemini image generation controversy, surprise first-ever dividend (8–10) |
| temporal_horizon | 1=days only → 10=structural multi-year shift | Single-day roundups (1) vs multi-year antitrust proceedings, decade-long nuclear partnerships, structural AI investments (8–10) |
| sentiment_strength | 1=neutral/balanced → 10=extremely strong positive or negative | Routine partnership announcement (2) vs $2T valuation/first dividend euphoria, DOJ breakup threat alarm (9–10) |
| information_density | 1=vague opinion → 10=specific financials, deal values, measurable outcomes | Trending alerts (1) vs detailed earnings with EPS, margin, segment data plus deal values (9–10) |
| directional_clarity | 1=ambiguous → 10=unambiguously clear direction | Events where competitive benefit and regulatory risk coexist (2–3) vs clear earnings beats, definitive fine amounts (9–10) |
| scope | 1=company only → 5=sector-wide → 10=macro/market-wide | CFO appointment (1–2) vs DMA investigations affecting all Big Tech (5–6) vs Fed policy or global tax negotiations (9–10) |
| competitive_impact | 1=no competitive effect → 10=fundamentally reshapes competitive landscape | Routine enterprise partnership (2) vs Gemini directly competing with OpenAI, Character.AI talent acquisition, DOJ search remedies threatening core moat (8–10) |
| regulatory_risk | 1=no regulatory angle → 10=existential threat such as forced structural separation | Product launches (1) vs DOJ seeking full corporate breakup, EU €2.7B fine upheld, UK CMA findings, simultaneous DMA non-compliance across jurisdictions (9–10) |
| expected_duration | 1=forgotten within one news cycle → 10=persistent for weeks/months with follow-up | One-day roundup mention (1) vs DOJ antitrust saga, EU DMA investigations, recurring Waymo safety concerns spanning full corpus period (9–10) |
| narrative_shift | 1=reinforces prevailing thesis → 10=fundamentally reframes investment thesis | Data center articles reinforcing established AI-capex narrative (1–2) vs DOJ breakup proposal and monopoly ruling reframing GOOGL as facing structural regulatory risk (9–10) |
| repeatedness | 1=nth update of known story → 10=first report of new information | Fifth EU fine update from multiple outlets (1) vs first-break Wiz acquisition talks, Gemini launch, CFO hire (9–10) |
| financial_result_surprise | 1=no results or in-line → 10=dramatic beat/miss | Non-earnings articles (1) vs Q4 2023 cloud disappointment causing selloff, Q1 2024 61% EPS growth + first-ever dividend, Q3 2024 cloud acceleration beat (6–10) |

### Dropped Dimensions

| Dimension | Dropped by | Reason |
|-----------|-----------|--------|
| management_signal | 2/3 runs | Would score uniformly low for ~90% of articles since most GOOGL coverage is regulatory/product/market, not leadership-driven; few leadership articles covered by leadership_workforce_corporate + materiality |
| controversy | 2/3 runs | GOOGL investors rarely show genuine two-sided division; antitrust and AI competition are broadly agreed-upon concerns where they differ on magnitude not direction; already captured by regulatory_risk + directional_clarity |
| actionability | 3/3 runs | Mega-cap with deep liquidity means nearly all substantive news is immediately actionable; insufficient variation to discriminate |

---

## META — Meta Platforms, Inc.

### Categories (9 total)

| ID | Label | Core Coverage |
|----|-------|---------------|
| earnings_financial_results | Earnings & Financial Results | Quarterly EPS/revenue vs expectations, dividends, capex forecasts, buybacks |
| market_sector_sentiment | Market & Sector Sentiment | Broad market roundups, macro data, index movements, multi-stock previews where META is secondary |
| analyst_consensus_signals | Analyst & Consensus Signals | Upgrades, downgrades, price targets, institutional positioning commentary |
| ai_strategy_infrastructure | AI Strategy & Infrastructure | Llama, Movie Gen, Meta AI assistant, AI infrastructure investments, custom chips, data center buildouts, energy deals, open-source AI strategy |
| regulatory_antitrust_compliance | Regulatory, Antitrust & Compliance | EU DMA/DSA, antitrust fines, FTC actions, GDPR enforcement, data privacy rulings, competition probes |
| child_safety_content_moderation | Child Safety & Content Moderation | Youth social media lawsuits and legislation, teen account protections, sextortion enforcement, Senate hearings on child exploitation |
| hardware_metaverse_ar_vr | Hardware, AR/VR & Metaverse | Quest headsets, Orion AR glasses, Reality Labs, EssilorLuxottica smart eyewear, mixed reality ecosystem |
| platform_product_updates | Platform & Product Updates | Feature launches across FB, Instagram, WhatsApp, Threads, Messenger; creator tools, ad capabilities, music deals, outages |
| geopolitical_government_relations | Geopolitical & Government Relations | Country-specific bans (Turkey, Russia, Brazil), content removal requests, election integrity, foreign interference, political content decisions |

**Why 9 categories?** META's uniqueness is the child safety category — no other ticker has a sustained, legislatively-driven reputational and legal risk story of this nature. The Orion AR glasses warrant their own hardware category distinct from platform because Reality Labs represents a multi-year, multi-billion-dollar bet being evaluated separately by investors. The geopolitical category is unique to META because of its platform's role in state-level information operations.

### Dimensions (14 total)

| Dimension | Consensus | Scale Description | Key Discriminating Examples |
|-----------|-----------|------------------|-----------------------------|
| materiality | 3/3 | 1=routine update with no revenue implications → 10=directly impacts revenue/cost structure/TAM | Form 4 filings, minor features (1–2) vs $1.4B settlements, major earnings capex guidance (8–10) |
| surprise | 3/3 | 1=fully expected → 10=completely unexpected contradiction of consensus | Scheduled product launches (1) vs first-ever dividend, 16% plunge on unexpected capex guidance, $1.4B Texas settlement (8–10) |
| temporal_horizon | 3/3 | 1=transient days → 10=structural multi-year shift | One-day service outages (1) vs EU DMA compliance model, open-source AI strategy, Orion decade-long hardware strategy (8–10) |
| sentiment_strength | 3/3 | 1=neutral → 10=extremely strong positive or negative | Neutral corporate blog posts, Form 4 filings (1) vs child exploitation hearings (negative 9), tripled earnings + dividend initiation (positive 9) |
| information_density | 3/3 | 1=pure opinion → 10=hard data, specific figures, named sources | One-line headline fragments (1) vs earnings transcripts with specific EPS, margin, price target data (9–10) |
| directional_clarity | 3/3 | 1=ambiguous → 10=unambiguously good or bad | AI spending increases (2–4, genuinely ambiguous across time horizons) vs earnings beat and dividend (clear positive), antitrust fines (clear negative) (8–10) |
| scope | 3/3 | 1=META only → 5=sector-wide → 10=broad macro | Threads feature update (1) vs Big Tech DMA regulatory crackdowns (5–6) vs broad macro market selloff (9–10) |
| regulatory_risk | 3/3 | 1=no regulatory content → 10=major EU/FTC action with direct financial consequences | Product announcements (1) vs major EU DMA charges, $1.4B Texas settlement, FTC privacy actions, multiple country investigations (8–10) |
| competitive_impact | 3/3 | 1=no competitive angle → 10=major competitive shift | Routine updates (1) vs open-sourcing Llama 3 (competitive moat builder), EssilorLuxottica partnership, Movie Gen rivaling OpenAI's Sora (7–9) |
| management_signal | 2/3 | 1=no leadership angle → 10=major CEO-level strategic announcement | Routine news (1) vs Zuckerberg Senate apology, AI team reorganization, open-source vision letters, Reality Labs restructuring (8–10) |
| narrative_shift | 3/3 | 1=reinforces existing narrative → 10=fundamentally challenges investment thesis | Most articles reinforce existing narratives (1–3) vs first dividend signaling capital return era, Q1 capex guidance sparking AI spending fears, Orion reframing hardware ambitions (7–9) |
| repeatedness | 3/3 | 1=nth rehash → 10=first report of new information | Multi-outlet coverage of same events (1) vs first-break dividend announcement, Orion reveal, Texas settlement (9–10) |
| financial_result_surprise | 3/3 | 1=no results or in-line → 10=dramatic beat/miss vs expectations | Non-earnings articles (1) vs Q4 2023 EPS tripled expectations, Q1 2024 guidance disappointment causing 16% plunge (8–10) |
| expected_duration | 2/3 | 1=resolves within a day → 10=persistent story for weeks/months | Differentiates temporal impact (horizon) from news-cycle persistence; regulatory ruling may have years-long horizon but fade from headlines in days |

### Dropped Dimensions

| Dimension | Dropped by | Reason |
|-----------|-----------|--------|
| actionability | 3/3 runs | Clusters at extremes; largely duplicates materiality without adding discriminating power |
| controversy | 3/3 runs | Most META articles reflect consensus views; few divisive items overlap heavily with directional_clarity |
| strategic_signal | — | Custom dimension from run 2 only (1/3 runs); concept absorbed into management_signal which was kept by 2/3 runs |

---

## MSFT — Microsoft Corporation

### Categories (9 total)

| ID | Label | Core Coverage |
|----|-------|---------------|
| ai_cloud_infrastructure_investment | AI & Cloud Infrastructure Investment | Global data center buildouts, multi-billion capex, nuclear/renewables energy procurement, regional capacity expansion |
| openai_partnership_dynamics | OpenAI Partnership & Governance | Altman firing/reinstatement saga, funding rounds, board dynamics, structural changes to OpenAI, talent flows, enterprise competitive dynamics |
| antitrust_regulatory_compliance | Antitrust, Regulatory & Legal | EU Teams bundling charges, CISPE cloud licensing complaints, FTC/CMA OpenAI investment probes, UK Inflection scrutiny, LinkedIn GDPR fine |
| cybersecurity_service_disruptions | Cybersecurity & Service Reliability | Midnight Blizzard Russian hacks, global CrowdStrike-linked outage, Azure/M365 service disruptions, government security review criticism, congressional testimony |
| strategic_partnerships_products | Strategic Partnerships & AI Products | Enterprise partnerships (Vodafone, Siemens, Palantir, BlackRock), Copilot launches, AI PC hardware, autonomous agents, Azure Marketplace integrations |
| gaming_entertainment_strategy | Gaming & Entertainment Strategy | Xbox multi-platform strategy, Activision Blizzard integration, CoD subscription shift, next-gen console planning, gaming layoffs, Sony/Nintendo competition |
| earnings_financial_results | Earnings & Financial Results | Quarterly EPS/revenue, cloud guidance revisions, $60B buyback, dividends, securities litigation tied to disclosures |
| market_sector_sentiment | Market & Sector Sentiment | Tech rallies/selloffs, Nasdaq corrections, Fed rate impacts, AI speculation waves, Magnificent Seven narratives |
| analyst_consensus_signals | Analyst & Consensus Signals | Upgrades/downgrades, price targets, initiations from Wedbush, UBS, HSBC, BofA, New Street Research |

**Why 9 categories?** MSFT has a category unique in the corpus: `openai_partnership_dynamics`. The Altman firing was a singular corporate governance event that created ongoing uncertainty about MSFT's most important strategic asset. The cybersecurity category is also unique — no other ticker faced a government security review board report combined with a multi-country state-sponsored hack and a global outage simultaneously.

### Dimensions (14 total — consensus 3/3)

| Dimension | Scale Description | Key Discriminating Examples |
|-----------|------------------|-----------------------------|
| materiality | 1=trivial operational detail → 10=reshaping a major revenue stream/cost structure | Small Azure Marketplace listings, carbon credit purchases (1–2) vs multi-billion infrastructure commitments, $100B Stargate plans, quarterly earnings (8–10) |
| surprise | 1=fully reflected in consensus → 10=completely unanticipated | Routine scheduled earnings, predictable dividends (1) vs Altman's firing, Russian hack escalation, CrowdStrike-linked global outage (9–10) |
| temporal_horizon | 1=confined to days → 10=structural multi-year shift | Service outages resolving in hours (1) vs EU antitrust charges, nuclear power purchase agreements, structural AI infrastructure buildouts (8–10) |
| sentiment_strength | 1=neutral/factual → 10=extremely strong positive or negative | Marketplace availability announcements, routine dividends (1–2) vs strongly negative security breach disclosures/government review reports, strongly positive earnings beats (8–10) |
| information_density | 1=pure opinion → 10=dense with specific financial figures, contractual terms | One-line flash headlines (1) vs earnings releases with EPS, revenue, margins, segment breakdowns, and forward guidance (9–10) |
| directional_clarity | 1=ambiguous → 10=unambiguously clear direction | OpenAI governance reshuffling (2–4, genuinely ambiguous) vs earnings beats (clearly positive), security breaches (clearly negative) (8–10) |
| scope | 1=company-specific → 5=sector-wide → 10=macro/market-wide | Xbox layoffs, Teams unbundling (1–2) vs sector-wide AI capex trends (5–6) vs Fed rate decisions, Nasdaq corrections (9–10) |
| competitive_impact | 1=no competitive shift → 10=major competitive advantage change | Routine partnership announcements (2) vs Google filing EU complaints against MSFT cloud practices, OpenAI pitching enterprise directly, Mistral diversification (6–8) |
| regulatory_risk | 1=no regulatory angle → 10=structural business changes or large penalties | Many articles (1) vs EU Teams bundling charges, CISPE settlement, UK Inflection probe, FTC OpenAI inquiry, LinkedIn GDPR fine, congressional security testimony (7–10) |
| management_signal | 1=no leadership angle → 10=major strategic pivot or C-suite decision signaling new direction | Routine product announcements (1) vs hiring DeepMind co-founder Suleyman to lead AI, Altman firing/reinstatement saga, MSFT leaving OpenAI's board, Brad Smith congressional testimony (9–10) |
| expected_duration | 1=transient, resolved in one news cycle → 10=persistent for weeks/months | CrowdStrike outage dominated one day then faded (2) vs Russian hacker intrusion with months of follow-up, EU antitrust proceedings generating articles spanning full corpus (8–10) |
| narrative_shift | 1=reinforces AI-leader thesis → 10=fundamentally challenges or reframes narrative | Most articles reinforce dominant AI-leader thesis (1–3) vs security review board report shifting to security-negligent framing, cloud revenue forecast cut challenging AI monetization story, Altman firing reframing OpenAI as liability (7–9) |
| repeatedness | 1=nth rehash → 10=first report of new information | 5–10 near-duplicate articles on same event (OpenAI $6.6B round, Three Mile Island deal, CISPE settlement) (1) vs first-break reports (9–10) |
| financial_result_surprise | 1=no results or in-line → 10=dramatic beat/miss | Most non-earnings articles (1) vs quarterly earnings cycles with varying beats/misses, cloud guidance revision as specific negative surprise, $60B buyback + dividend increase as concrete positive (6–10) |

### Dropped Dimensions

| Dimension | Dropped by | Reason |
|-----------|-----------|--------|
| actionability | 3/3 runs | Overlaps heavily with materiality + information_density; edge cases better captured by temporal_horizon |
| controversy | 3/3 runs | Most MSFT articles reflect consensus-aligned views; even antitrust/security stories generated limited investor disagreement (magnitude debate not direction debate) |
| geopolitical_sensitivity | 2/3 runs | Appeared in only 1/3 runs (Run 2); G42 scrutiny and China staff relocation adequately captured by regulatory_risk + scope + competitive_impact combination |

---

## NVDA — NVIDIA Corporation

### Categories (9 total)

| ID | Label | Core Coverage |
|----|-------|---------------|
| earnings_financial_results | Earnings & Financial Results | Quarterly EPS/revenue, margins, guidance, stock split, dividends, credit rating upgrades, results vs expectations |
| market_sector_sentiment | Market & Sector Sentiment | Mag-7 commentary, sector rotation, macro indicators (Fed, recession, inflation), index milestones, broad market roundups |
| analyst_consensus_signals | Analyst & Consensus Signals | Price target changes, upgrades/downgrades, ratings, institutional positioning disclosures, sell-side research |
| ai_platform_chip_launches | AI Platform & Chip Launches | Blackwell/Hopper architecture announcements, GTC/CES/Computex reveals, NIM microservices, DGX systems, networking hardware, benchmarks |
| regulatory_legal_geopolitical | Regulatory, Legal & Geopolitical Risk | DOJ antitrust, French competition authority charges, US-China export controls, securities fraud lawsuits, PSLRA Supreme Court case, copyright claims |
| enterprise_industry_partnerships | Enterprise & Industry Partnerships | Hyperscaler partnerships (AWS, Azure, Google Cloud, Oracle), SAP/Cisco/ServiceNow/Accenture integrations, healthcare/AV/robotics/telecom AI deployments, Run:ai acquisition |
| gaming_consumer_products | Gaming & Consumer Products | GeForce NOW additions, RTX GPU launches (40 SUPER), Gamescom, DLSS/RTX video technology, streaming partnerships |
| global_expansion_sovereign_ai | Global Expansion & Sovereign AI | Country-level sovereign AI supercomputers (India, Japan, Vietnam, Indonesia, Denmark, Thailand), government partnerships, international AI summits |
| competitive_landscape | Competitive Landscape & Moat | Huawei/AMD chip challenges, CUDA ecosystem lock-in debates, competitor launches, custom chip unit, market share dynamics |

**Why 9 categories?** NVDA has two categories unique in the corpus: `global_expansion_sovereign_ai` (reflecting an entirely new national AI investment trend that no other ticker triggered) and `competitive_landscape` as a standalone category (because NVDA's primary investor risk is whether its moat holds, making competitive positioning a material standalone theme). The gaming category is retained because NVDA operates dual B2B and B2C GPU businesses that react to entirely different market forces.

### Dimensions (14 total)

| Dimension | Consensus | Scale Description | Key Discriminating Examples |
|-----------|-----------|------------------|-----------------------------|
| materiality | 3/3 | 1=no revenue/margin impact → 10=directly affects near-term revenue, margins, or TAM | Weekly GeForce NOW game additions, blog posts (1) vs Blackwell platform launches defining next-gen data center revenue, quarterly earnings, DOJ subpoenas (8–10) |
| surprise | 3/3 | 1=fully expected → 10=completely unexpected relative to prevailing assumptions | Pre-announced GTC keynotes, scheduled weekly posts (1) vs DOJ subpoena reports, Blackwell production delay leaks, Q2 FY25 beat that disappointed elevated expectations (8–10) |
| temporal_horizon | 3/3 | 1=days only → 10=structural multi-year shift | Single-day game launches, weekly recaps (1) vs Blackwell architecture, export control policy shifts, antitrust investigations (8–10) |
| sentiment_strength | 3/3 | 1=neutral technical blogs → 10=extremely strong positive/negative | Neutral blog posts, informational product descriptions (1) vs global rallies on earnings with superlatives "soaring"/"stunning", antitrust probe "plunged" coverage (9–10) |
| information_density | 3/3 | 1=pure opinion → 10=earnings with exact figures, analyst targets with precise price points | Opinion columns, market-mood pieces (1) vs earnings with revenue/EPS/margins, product launches with detailed technical specifications (9–10) |
| directional_clarity | 3/3 | 1=ambiguous → 10=clearly positive or negative | AI valuation debate articles (2–3, could go either way) vs clearly positive massive earnings beats, clearly negative subpoena reports (9–10) |
| scope | 3/3 | 1=NVDA specific → 5=semiconductor sector-wide → 10=macro/market-wide | Run:ai acquisition (1) vs semiconductor demand signals (4–5) vs recession fears, Fed rate decisions, global equity selloffs (9–10) |
| competitive_impact | 3/3 | 1=no competitive implication → 10=fundamentally reshapes competitive landscape | Routine partnership announcements (2) vs Huawei launching rival AI chips, AMD data center challenge to dominance, Silicon Valley break-CUDA efforts, France anticompetitive charges, custom chip unit targeting ASIC market (8–10) |
| regulatory_risk | 3/3 | 1=no regulatory angle → 10=major regulatory action with direct consequences | Product/partnership articles (1) vs DOJ antitrust probes, French competition charges, US-China export controls, Supreme Court securities fraud case (8–10) |
| expected_duration | 2/3 | 1=forgotten within a day → 10=persistent for months | Daily market roundups, single GeForce NOW additions (1) vs DOJ antitrust investigation, Blackwell production timeline, US-China export control evolution persisting across full corpus period (8–10) |
| narrative_shift | 3/3 | 1=reinforces AI-demand-insatiable thesis → 10=fundamentally challenges dominant narrative | Most articles reinforcing AI demand thesis (1–3) vs Blackwell delays, DOJ escalation, dot-com bubble comparisons, semiconductor glut fears, earnings that technically beat but disappointed (6–9) |
| repeatedness | 3/3 | 1=nth rehash → 10=first report of genuinely new information | Multiple outlets repackaging same earnings/France antitrust details (1) vs Reuters custom chip unit scoop, DOJ subpoena disclosure, Blackwell delay leak (9–10) |
| actionability | 3/3 | 1=educational background context → 10=time-sensitive information investors can act on immediately | AI educational blog posts, corporate thought leadership (1) vs earnings releases, analyst target changes, antitrust subpoena reports, chip delay disclosures (9–10) |
| financial_result_surprise | 3/3 | 1=no results or in-line → 10=dramatic beat/miss | Non-earnings articles (1) vs Q4 FY24 blowout (revenue +265%) sparking global rallies, Q1 FY25 beat triggering stock split, Q2 FY25 technical beat that fell short of elevated expectations (7–10) |

### Dropped Dimensions

| Dimension | Dropped by | Reason |
|-----------|-----------|--------|
| controversy | 3/3 runs | NVDA coverage was overwhelmingly consensus-directional; even antitrust/bubble debates generated one-sided investor reactions rather than genuine two-sided division |
| management_signal | 2/3 runs | Jensen Huang's presence is so pervasive across product launches, keynotes, and strategic commentary that most substantive articles carry some management signal; clusters around moderate values with insufficient discrimination |
| demand_signal_strength | — | Custom dimension from Run 1 only (1/3); concept absorbed into materiality + narrative_shift + competitive_impact |

---

## TSLA — Tesla, Inc.

### Categories (9 total)

| ID | Label | Core Coverage |
|----|-------|---------------|
| earnings_financial_results | Earnings & Financial Results | Quarterly earnings, delivery and production numbers, revenue, margins, EPS, guidance, energy storage, results vs expectations |
| market_sector_sentiment | Market & Sector Sentiment | Broad market roundups, sector rotation, macro data, EV or tech industry sentiment mentioning TSLA in wider narrative |
| analyst_consensus_signals | Analyst & Consensus Signals | Upgrades/downgrades, price targets, proxy advisor recommendations, institutional positioning |
| autonomous_driving_robotaxi | Autonomous Driving & Robotaxi | FSD development, robotaxi strategy, Cybercab unveil, NHTSA Autopilot investigations, FSD regulatory approvals, safety incidents |
| musk_compensation_governance | Musk Compensation & Governance | $56B pay package litigation, Delaware court ruling, reincorporation to Texas, board independence, demands for greater voting control |
| pricing_demand_competition | Pricing, Demand & Competition | Vehicle price cuts/increases globally, demand signals, incentive programs, EV price wars, regional monthly sales, market share shifts |
| recalls_safety_regulatory | Recalls, Safety & Regulatory | Vehicle recalls across all models, NHTSA safety investigations, steering/suspension probes, crash investigations, environmental compliance |
| workforce_operations | Workforce & Operations | Mass layoffs/restructuring, executive departures, factory shutdowns, gigafactory expansion, union organizing, supercharger team changes |
| musk_political_legal_personal | Musk Political, Legal & Personal | Political endorsements, campaign activity, SEC/DOJ enforcement, personal controversies, lawsuits from other ventures, conduct reporting creating headline risk |

**Why 9 categories?** TSLA is the only ticker that requires two separate Musk-specific categories. The governance/compensation story (`musk_compensation_governance`) is a discrete legal and structural saga with direct valuation implications through dilution and incentive alignment. The political/personal category (`musk_political_legal_personal`) captures a different risk — reputational and brand contagion effects from non-Tesla activities, including association with political figures that caused consumer boycott risk and brand damage. No other ticker in the corpus has this duality.

### Dimensions (15 total — richest schema)

| Dimension | Consensus | Scale Description | Key Discriminating Examples |
|-----------|-----------|------------------|-----------------------------|
| materiality | 3/3 | 1=trivial tangential mention → 10=directly alters revenue trajectory, margin structure, or TAM | Casual market roundup mentions, personal anecdotes (1) vs cancellation of major vehicle platform, quarterly earnings revealing margin compression, workforce reductions, global pricing strategy changes (8–10) |
| surprise | 3/3 | 1=fully anticipated → 10=complete shock contradicting prevailing expectations | Scheduled events meeting consensus (1) vs major earnings beats after quarters of misses, abrupt product line cancellations, factory arson attacks, executive departures (8–10) |
| temporal_horizon | 3/3 | 1=days only → 10=structural multi-year shift | Single-day price adjustments, one-off recall notices (1) vs abandoning a product line for new business model, regulatory regime changes reshaping competitive dynamics for years, ongoing compensation litigation saga (8–10) |
| sentiment_strength | 3/3 | 1=neutral/flat tone → 10=intensely charged language conveying alarm, euphoria, or combativeness | Data releases, neutral regulatory filings (1) vs alarm over safety fatalities, euphoria over earnings surprises, combative executive rhetoric toward advertisers and critics (9–10) |
| information_density | 3/3 | 1=pure opinion columns → 10=specific EPS, delivery counts, margins, energy deployment, precise price changes | Speculative analysis (1) vs articles packed with specific delivery counts, margin percentages, energy deployment numbers, precise price changes across multiple markets (9–10) |
| directional_clarity | 3/3 | 1=ambiguous → 10=unambiguously positive or negative | Tariff structures, executive control demands, business model pivots (2–4, genuinely ambiguous splitting investor interpretation) vs confirmed investigations, earnings beats (8–10) |
| scope | 2/3 | 1=company only → 10=macro/market-wide | Specific model recalls, internal layoff notices (1) vs sector-wide EV price wars, tariff regimes (4–6) vs monetary policy, broad index rotation (9–10) |
| competitive_impact | 2/3 | 1=no competitive shift → 10=fundamentally reshapes competitive standing | Routine operational updates (1) vs rival companies overtaking in global sales, competitor bankruptcies removing challengers, tariff structures reshaping import economics (7–9) |
| regulatory_risk | 3/3 | 1=no regulatory angle → 10=major regulatory action with direct operational/liability consequences | Articles with zero regulatory content (1) vs multiple active safety investigations following fatalities, securities enforcement against executives, fraud examinations of product claims, tariff determinations (8–10) |
| management_signal | 3/3 | 1=routine update → 10=executive demands, strategic pivots redefining company direction | Routine operational updates (1) vs executive demands for greater voting control before advancing key initiatives, senior leadership departures, entire team eliminations, strategic pivots redefining company direction (9–10) |
| narrative_shift | 3/3 | 1=reinforces existing narrative → 10=challenges or reverses prevailing investment thesis | Routine news reinforcing existing narratives (1–3) vs product line cancellations in favor of new business model, earnings turnarounds after deterioration, executive political entanglements reversing prevailing thesis (7–10) |
| repeatedness | 3/3 | 1=nth rehash → 10=first-break report of new events | Explicit rehashes of identical figures, boilerplate legal notices, nth article on ongoing saga (1) vs first-break reports of genuinely new events (9–10) |
| actionability | 2/3 | 1=background color or historical context → 10=immediately actionable concrete new information | Background industry context pieces (1) vs revised quarterly guidance, confirmed price changes effective on specific dates, delivery numbers vs consensus, recall announcements with specific vehicle counts (9–10) |
| financial_result_surprise | 3/3 | 1=no results or in-line → 10=dramatic beat/miss | Most non-earnings articles (1) vs earnings misses with margin compression, delivery misses vs consensus, modest beats against lowered expectations, dramatic upside surprises (5–10) |
| controversy | 2/3 | 1=consensus-view reports → 10=deeply polarizing event dividing investors and analysts | Consensus-view delivery reports (1) vs divisive events where major institutional/retail shareholders hold sharply opposing views, executive actions polarizing investor base, product safety debates dividing analysts (7–10) |

### Dropped Dimensions

| Dimension | Dropped by | Reason |
|-----------|-----------|--------|
| expected_duration | 2/3 runs | Appeared in only 1/3 runs; both Run 1 and Run 2 independently dropped it as too correlated with temporal_horizon — events with long-term structural impact also tend to persist in news cycles |

---

## Cross-Ticker Comparative Analysis

### Category Convergence — Universal vs Ticker-Specific

**Present in all 7 tickers:**

| Category Theme | Notes |
|----------------|-------|
| Earnings & Financial Results | Universal — every ticker's primary valuation anchor |
| Market & Sector Sentiment | Universal — all are large-cap S&P constituents subject to macro flows |
| Analyst & Consensus Signals | Universal — all covered by major sell-side desks |

**Present in 5/7 tickers:**

| Category Theme | Which Tickers | Notes |
|----------------|--------------|-------|
| AI Strategy / AI Products | AAPL, AMZN, GOOGL, META, MSFT | NVDA is itself the AI infrastructure; TSLA's FSD is autonomous-driving-specific |
| Cloud / Infrastructure Investment | AMZN (AWS), GOOGL, MSFT, NVDA (sovereign AI), indirectly META | AAPL and TSLA don't have cloud categories |

**Ticker-exclusive categories:**

| Ticker | Unique Category | Rationale |
|--------|----------------|-----------|
| AAPL | china_market_competition | Only ticker with >20% revenue concentration in a single geopolitically contested market |
| AAPL | supply_chain_manufacturing | Uniquely complex global manufacturing dependency among the 7 |
| AMZN | content_entertainment_sports | Only ticker operating a major content studio + sports broadcaster |
| AMZN | labor_workforce_organization | Most intense and persistent labor organizing activity among the 7 |
| META | child_safety_content_moderation | Unique legislatively-driven reputational/legal risk |
| META | geopolitical_government_relations | Platform used in state-level information operations uniquely |
| MSFT | openai_partnership_dynamics | Unique structural dependency on a single AI partner |
| MSFT | cybersecurity_service_disruptions | Only ticker combining a government security review with a state-sponsored hack and global outage simultaneously |
| NVDA | global_expansion_sovereign_ai | Triggered an entirely new national AI investment trend |
| NVDA | competitive_landscape | Primary investor risk is moat durability; warranted standalone category |
| TSLA | autonomous_driving_robotaxi | FSD/robotaxi is the primary long-horizon value driver with independent investor base |
| TSLA | musk_compensation_governance | Unique ongoing legal saga with direct dilution implications |
| TSLA | musk_political_legal_personal | Unique CEO personal brand / reputational contagion risk |
| TSLA | pricing_demand_competition | Uniquely aggressive pricing strategy creating ongoing demand-signal ambiguity |
| TSLA | recalls_safety_regulatory | Uniquely high volume of NHTSA interactions and active safety investigations |

---

### Dimension Convergence — What Every Ticker Agrees On

**Kept by all 7 tickers (3/3 consensus within each ticker):**

| Dimension | Universal Rationale |
|-----------|---------------------|
| materiality | Every ticker shows wide variation from trivial to market-moving events |
| surprise | Every ticker had genuinely unexpected events in the corpus window |
| temporal_horizon | Every ticker spans from one-day events to multi-year structural shifts |
| sentiment_strength | Every ticker's corpus has neutral articles and strongly valenced articles |
| information_density | Every ticker has opinion pieces and data-rich earnings reports |
| directional_clarity | Every ticker has clearly directional and genuinely ambiguous events |
| narrative_shift | Every ticker had at least one thesis-changing event in the corpus |
| repeatedness | Every ticker's corpus contains both first-break stories and nth rehashes |
| financial_result_surprise | Every ticker reported at least 3 quarterly earnings cycles with meaningful beat/miss variation |

**Kept by most but not all tickers:**

| Dimension | Absent In | Notes |
|-----------|----------|-------|
| regulatory_risk | — | All 7 kept. TSLA regulatory risk uniquely includes safety + securities enforcement |
| competitive_impact | — | All 7 kept. NVDA/TSLA had strongest variation |
| scope | — | All 7 kept. TSLA kept in 2/3 sub-runs but included in consensus |
| management_signal | NVDA | NVDA dropped because Jensen Huang's omnipresence diluted discriminating power |
| expected_duration | AMZN, TSLA | AMZN: universally judged redundant with temporal_horizon. TSLA: only 1/3 runs kept it |

---

### Dropped Dimensions — What the LLMs Consistently Rejected

| Dimension | Dropped By | Why it failed |
|-----------|-----------|---------------|
| actionability | 5/7 tickers (AAPL, AMZN, GOOGL, META, MSFT); kept only by NVDA (3/3) and TSLA (2/3) | For mega-caps with deep liquidity, most substantive news is immediately actionable, causing it to cluster high and lose discriminating power. NVDA/TSLA retained it because their corpus includes genuinely non-actionable educational and background-context content |
| controversy | 6/7 tickers (all except TSLA) | Most Big Tech investor bases hold consensus-aligned views; even regulatory and antitrust stories generate magnitude debate not directional disagreement. TSLA retained it because Musk-related events genuinely polarize institutional and retail shareholders in opposite directions |
| expected_duration | 3/7 tickers (AMZN, TSLA, NVDA dropped); kept by AAPL, GOOGL, META, MSFT | When retained, it captures news-cycle persistence independently of structural impact (a regulation may have long-term horizon but fade from headlines in days); when dropped, the corpus didn't show enough distinction between duration and horizon |

---

### Schema Size Comparison

| Ticker | Categories | Dimensions (final) | Dimensions Dropped | Notable |
|--------|-----------|--------------------|--------------------|---------|
| AAPL | 9 | 14 | 2 | expected_duration retained |
| AMZN | 9 | 13 | 3 | Leanest schema; expected_duration 3/3 dropped |
| GOOGL | 9 | 13 | 3 | Leanest schema; expected_duration retained |
| META | 9 | 14 | 3 | management_signal at 2/3 only |
| MSFT | 9 | 14 | 3 | All 14 retained dims are 3/3 |
| NVDA | 9 | 14 | 3 | Only ticker retaining actionability (3/3) |
| TSLA | 9 | 15 | 1 | Richest schema; only ticker retaining controversy |

TSLA has the richest dimension set (15) because its news is genuinely the most multi-dimensional: a CEO who functions as a narrative variable, a governance saga with legal stakes, product safety, political entanglement, and volatile demand signals all running simultaneously. AMZN and GOOGL have the leanest schemas (13 dimensions) because several candidate dimensions were redundant given the structure of their news corpus.

---

### Summary Table — Final Schemas

| Dimension | AAPL | AMZN | GOOGL | META | MSFT | NVDA | TSLA |
|-----------|------|------|-------|------|------|------|------|
| Categories | 9 | 9 | 9 | 9 | 9 | 9 | 9 |
| Dimensions | 14 | 13 | 13 | 14 | 14 | 14 | 15 |
| materiality | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| surprise | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| temporal_horizon | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| sentiment_strength | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| information_density | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| directional_clarity | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| scope | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| competitive_impact | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| regulatory_risk | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| management_signal | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| expected_duration | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ |
| narrative_shift | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| repeatedness | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| financial_result_surprise | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| actionability | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| controversy | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |

*✓ = included in final schema | ✗ = dropped*

---

## Key Findings for Stage 3 Model Design

**1. A shared 12-dimension core scoring vector works cross-ticker.** Materiality, surprise, temporal_horizon, sentiment_strength, information_density, directional_clarity, scope, competitive_impact, regulatory_risk, narrative_shift, repeatedness, and financial_result_surprise are present in every ticker's final schema.

**2. Ticker-specific dimensions require masking or zero-imputation.** `actionability` (NVDA/TSLA only) and `controversy` (TSLA only) should be optional features with ticker-level handling in the downstream model.

**3. financial_result_surprise is the most reliable separator.** It scores 1 for ~80–90% of articles and mid-to-high only for earnings-cycle events. Within earnings events it discriminates beat/miss magnitude. This ensures the model treats earnings articles as a distinct class.

**4. Categories are non-transferable.** Each ticker's 9-category taxonomy is entirely specific to its news landscape. There is no universal category ontology — the 3-run consensus process produced substantively different taxonomies even for closely related companies (MSFT vs GOOGL, META vs AAPL).

**5. The actionability exception at NVDA/TSLA is meaningful.** These are the two tickers with the widest variation between genuine background-context articles (educational AI posts, historical industry context) and immediately actionable information (delivery counts, subpoena disclosures, chip delay reports). The dimension earns its place only where there is real discrimination to be had.

**6. The controversy exception at TSLA is data-driven.** Musk-related events uniquely split institutional and retail shareholders in opposite directions. For the other 6 tickers, investor bases hold consensus-aligned views even on negative stories — they disagree on magnitude, not direction. Controversy collapses to directional_clarity for those tickers.

**7. TSLA is categorically different in news structure.** 15 dimensions, two CEO-specific categories, and the only ticker where CEO activity is an independent news variable (not just a management signal). This will require careful feature engineering to avoid the model over-indexing on Musk-specific patterns.

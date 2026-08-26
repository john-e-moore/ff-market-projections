-- DuckDB-compatible rendering views for the validated report datasets.

-- The joined calculations are executed and documented in projection_audit.ipynb.

CREATE OR REPLACE VIEW "headline" AS
SELECT * FROM (VALUES
  (0, 0.4430379746835443, 0.8544303797468354, 171)
) AS dataset("kalshi_updates", "kalshi_disagreement_rate", "inactive_curve_rate", "blank_names");

CREATE OR REPLACE VIEW "pair_by_stat" AS
SELECT * FROM (VALUES
  ('DraftKings vs FanDuel', 'Passing Touchdowns', 'DK vs FD · Passing Touchdowns', 21, 0.027333300789567867, 0.9877683851462554),
  ('DraftKings vs FanDuel', 'Passing Yards', 'DK vs FD · Passing Yards', 21, 0.02444392947640358, 0.9849214865494088),
  ('DraftKings vs FanDuel', 'Receiving Yards', 'DK vs FD · Receiving Yards', 40, 0.0242998907908706, 1.0008516968536716),
  ('DraftKings vs FanDuel', 'Rushing Touchdowns', 'DK vs FD · Rushing Touchdowns', 20, 0.0678604737469409, 1.011161885679076),
  ('DraftKings vs FanDuel', 'Rushing Yards', 'DK vs FD · Rushing Yards', 25, 0.025866078427367004, 0.976088915225835),
  ('DraftKings vs Kalshi', 'Passing Touchdowns', 'DK vs Kalshi · Passing Touchdowns', 1, 0.2929464794563817, 0.7444803164129375),
  ('DraftKings vs Kalshi', 'Passing Yards', 'DK vs Kalshi · Passing Yards', 5, 0.3469444048726126, 0.7043437380516527),
  ('DraftKings vs Kalshi', 'Receiving Touchdowns', 'DK vs Kalshi · Receiving Touchdowns', 20, 0.14346026362886166, 0.866178209369739),
  ('DraftKings vs Kalshi', 'Receiving Yards', 'DK vs Kalshi · Receiving Yards', 25, 0.15333732863340038, 0.8663517649800859),
  ('DraftKings vs Kalshi', 'Receptions', 'DK vs Kalshi · Receptions', 17, 0.28678072601825394, 0.7491838874139918),
  ('DraftKings vs Kalshi', 'Rushing Touchdowns', 'DK vs Kalshi · Rushing Touchdowns', 15, 0.2807621982716703, 1.1426536347245102),
  ('DraftKings vs Kalshi', 'Rushing Yards', 'DK vs Kalshi · Rushing Yards', 17, 0.3692923178297664, 0.6882678299754652),
  ('FanDuel vs Kalshi', 'Passing Touchdowns', 'FD vs Kalshi · Passing Touchdowns', 1, 0.2724850741262639, 0.7601875785863805),
  ('FanDuel vs Kalshi', 'Passing Yards', 'FD vs Kalshi · Passing Yards', 4, 0.4164482955532013, 0.6575260115041439),
  ('FanDuel vs Kalshi', 'Receiving Yards', 'FD vs Kalshi · Receiving Yards', 15, 0.1683813494234105, 0.8446939700268272),
  ('FanDuel vs Kalshi', 'Rushing Touchdowns', 'FD vs Kalshi · Rushing Touchdowns', 12, 0.2049175679723006, 1.1125240387596622),
  ('FanDuel vs Kalshi', 'Rushing Yards', 'FD vs Kalshi · Rushing Yards', 15, 0.38273903394087466, 0.6787402829357669)
) AS dataset("pair", "stat", "comparison", "overlap", "median_abs_difference", "median_right_to_left_ratio");

CREATE OR REPLACE VIEW "kalshi_funnel" AS
SELECT * FROM (VALUES
  ('Raw Kalshi contracts', 1129),
  ('Two-sided and spread-eligible', 336),
  ('Quotes on curves used', 187),
  ('Eligible player/stat curves', 158),
  ('Stats with Kalshi dispersion update', 0)
) AS dataset("stage", "count");

CREATE OR REPLACE VIEW "triple_by_stat" AS
SELECT * FROM (VALUES
  ('Passing Touchdowns', 1, -0.2477480367328101, 0.2477480367328101, -0.08258267891008499, TRUE),
  ('Passing Yards', 3, -0.2889094250170809, 0.2889094250170809, -0.09630314167301791, 3),
  ('Receiving Yards', 13, -0.1441701368690224, 0.1441701368690224, -0.04805671228689472, 9),
  ('Rushing Touchdowns', 11, 0.1378980756590213, 0.26237014318822033, 0.04596602521934307, 8),
  ('Rushing Yards', 14, -0.3166959643437194, 0.3166959643437194, -0.10556532144705053, 14)
) AS dataset("stat_label", "overlaps", "median_kalshi_vs_books", "median_abs_kalshi_vs_books", "median_consensus_shift", "disagreement_flags");

CREATE OR REPLACE VIEW "triple_outliers" AS
SELECT * FROM (VALUES
  ('James Cook', 'Rushing Touchdowns', 12.1569194739, 12.184584908, 21.8931029163, 0.7988290758708735, 0.2662763586263633, 1, 119.72, 119.72, 0.0, 0.08, 5.29),
  ('Jahmyr Gibbs', 'Rushing Touchdowns', 15.1548171147, 16.2724865312, 25.1622686574, 0.6012998722964047, 0.20043329076758956, 1, 29.93, 29.93, 0.82, 0.09, 4.18),
  ('Travis Kelce', 'Receiving Yards', 851.584200823, 852.846445092, 498.675854024, -0.414847585357405, -0.13828252845246827, 1, 455.35, 455.35, 0.0, 0.18, 5.0),
  ('Christian McCaffrey', 'Rushing Yards', 1220.71614166, 1157.73021769, 696.66475483, -0.41418501864352336, -0.13806167288117446, 1, 0.0, 0.0, 0.0, 0.11, 200.0),
  ('Devon Achane', 'Rushing Yards', 1252.85181727, 1254.1372443, 736.299894631, -0.4126022279731107, -0.13753407599396217, 1, 54.44, 54.44, 0.0, 0.08, 10.0),
  ('Kyren Williams', 'Rushing Yards', 1220.71614166, 1189.86589315, 710.014441502, -0.41091866507835917, -0.13697288835724042, 1, 0.0, 0.0, 0.0, 0.1, 600.0),
  ('Omarion Hampton', 'Rushing Yards', 1220.71614166, 1189.86589315, 718.923718842, -0.40352685910673436, -0.13450895303613103, 2, 1100.97, 307.61, 0.0, 0.095, 5.0),
  ('Jaxson Dart', 'Passing Yards', 4027.20815877, 4068.42842801, 2429.1939527, -0.39987574129332304, -0.13329191376526445, 2, 294.88, 0.0, 213.6, 0.145, 5.0),
  ('Jeremiyah Love', 'Rushing Yards', 995.766414907, 997.051841902, 641.352253757, -0.35633643302328516, -0.11877881100809626, 1, 30.0, 30.0, 0.0, 0.09, 150.0),
  ('Travis Etienne', 'Rushing Yards', 1060.03776488, 1125.59454233, 729.589892126, -0.33237636566844586, -0.11079212188948188, 1, 328.75, 328.75, 0.0, 0.09, 5.0),
  ('Jahmyr Gibbs', 'Receiving Yards', 567.57925257, 600.397600349, 392.877662086, -0.3272509449067887, -0.10908364830226289, 1, 530.69, 530.69, 0.0, 0.17, 20.0),
  ('Josh Allen', 'Rushing Yards', 642.27400438, 643.559431255, 436.809710456, -0.32058119138847174, -0.10686039713000907, 1, 1.0, 1.0, 0.0, 0.12, 30.0),
  ('Bijan Robinson', 'Rushing Yards', 1509.93722418, 1479.08697516, 1027.01266787, -0.31281073729896697, -0.10427024576409197, 1, 719.6, 714.6, 420.0, 0.11, 90.0),
  ('James Cook', 'Rushing Yards', 1574.20857641, 1511.22265122, 1065.81692078, -0.3091293617335417, -0.1030431205800079, 1, 0.0, 0.0, 0.0, 0.14, 5.0),
  ('George Kittle', 'Receiving Yards', 883.140307679, 884.402551957, 620.125167261, -0.29831951301174575, -0.09943983767058195, 1, 291.8, 291.8, 0.0, 0.01, 58.0)
) AS dataset("canonical_player_name", "stat_label", "draftkings_mean", "fanduel_mean", "kalshi_mean", "kalshi_vs_books", "consensus_vs_books", "contributing_quotes", "total_volume", "minimum_open_interest", "volume_24h", "median_spread", "minimum_top_size");

CREATE OR REPLACE VIEW "output_quality" AS
SELECT * FROM (VALUES
  ('Blank player names', 171, 171, 'High', 'High'),
  ('Rows marked projection_complete=false', 171, 171, 'High', 'High'),
  ('Rows without a numeric standard score', 80, 171, 'Medium', 'High'),
  ('Unmatched source/player identities', 77, 363, 'Medium', 'High')
) AS dataset("issue", "affected", "total", "severity", "confidence");

CREATE OR REPLACE VIEW "top_scores" AS
SELECT * FROM (VALUES
  ('Josh Allen', 'QB', 'passing|rushing', 450.196316384, 450.196316384, 4740.23289161, 29.5336522348, 574.21438203, 14.1718255963, NULL, NULL, NULL),
  ('Drake Maye', 'QB', 'passing|rushing', 413.174182592, 413.174182592, 5089.67980141, 32.3031988093, 530.441870147, 4.55500138054, NULL, NULL, NULL),
  ('Jalen Hurts', 'QB', 'passing|rushing', 391.729677959, 391.729677959, 4198.9841782, 26.6400119126, 530.441870147, 10.6943460276, NULL, NULL, NULL),
  ('Lamar Jackson', 'QB', 'passing|rushing', 378.674179117, 378.674179117, 4231.78713052, 28.4253685343, 683.680166301, 4.55553385475, NULL, NULL, NULL),
  ('Jayden Daniels', 'QB', 'passing|rushing', 372.052702393, 372.052702393, 4133.37827358, 24.7390594661, 706.545349069, 6.18446644647, NULL, NULL, NULL),
  ('Joe Burrow', 'QB', 'passing', 353.542254242, 353.542254242, 5248.02259399, 35.9053376205, NULL, NULL, NULL, NULL, NULL),
  ('Jared Goff', 'QB', 'passing', 348.684411334, 348.684411334, 5209.95367044, 35.071566129, NULL, NULL, NULL, NULL, NULL),
  ('Jaxson Dart', NULL, 'passing|rushing', 339.964883997, 339.964883997, 3508.27684649, 23.2477903244, 594.713210712, 7.86188796144, NULL, NULL, NULL),
  ('Dak Prescott', 'QB', 'passing', 336.703010602, 336.703010602, 5248.67865304, 31.68896612, NULL, NULL, NULL, NULL, NULL),
  ('Jahmyr Gibbs', 'RB', 'receiving|rushing', 330.852833076, 389.429325243, NULL, NULL, 1404.79495202, 18.8631907678, 520.284838335, 4.19428490555, 58.5764921679),
  ('Brock Purdy', 'QB', 'passing', 328.027578895, 328.027578895, 4920.6491295, 32.8004034287, NULL, NULL, NULL, NULL, NULL),
  ('Matthew Stafford', 'QB', 'passing', 327.858093699, 327.858093699, 4580.17282168, 36.1627952079, NULL, NULL, NULL, NULL, NULL),
  ('Trevor Lawrence', 'QB', 'passing', 313.433953991, 313.433953991, 4873.06053806, 29.6278831171, NULL, NULL, NULL, NULL, NULL),
  ('Patrick Mahomes', 'QB', 'passing', 310.750367317, 310.750367317, 4788.78126107, 29.7997792185, NULL, NULL, NULL, NULL, NULL),
  ('Justin Herbert', 'QB', 'passing', 310.305103165, 310.305103165, 4755.74825684, 30.0187932229, NULL, NULL, NULL, NULL, NULL)
) AS dataset("canonical_player_name", "canonical_position", "scoring_profile", "fpts_standard", "fpts_full_ppr", "passing_yards", "passing_touchdowns", "rushing_yards", "rushing_touchdowns", "receiving_yards", "receiving_touchdowns", "receptions");

CREATE OR REPLACE VIEW "book_line_ratios" AS
SELECT * FROM (VALUES
  ('Draftkings', 'Passing Touchdowns', 28, 1.1662299634809317, 1.04707884288),
  ('Draftkings', 'Passing Yards', 28, 1.3119807480588286, 1.04761904762),
  ('Draftkings', 'Receiving Touchdowns', 55, 1.0637330291603717, 1.05555555556),
  ('Draftkings', 'Receiving Yards', 70, 1.2616885538467242, 1.04761904762),
  ('Draftkings', 'Receptions', 43, 1.1902550350087577, 1.05869324474),
  ('Draftkings', 'Rushing Touchdowns', 33, 1.1850199828743455, 1.05555555556),
  ('Draftkings', 'Rushing Yards', 37, 1.284877612289321, 1.04761904762),
  ('Fanduel', 'Passing Touchdowns', 23, 1.1409137094822677, 1.06896551724),
  ('Fanduel', 'Passing Yards', 23, 1.3119896215751574, 1.06542056075),
  ('Fanduel', 'Receiving Yards', 43, 1.261766273080772, 1.06542056075),
  ('Fanduel', 'Rushing Touchdowns', 21, 1.2456956044191905, 1.06878031878),
  ('Fanduel', 'Rushing Yards', 26, 1.28493223747753, 1.06542056075)
) AS dataset("source", "stat", "quotes", "median_mean_to_threshold", "median_overround");

-- Migration: Add 4 new entropy features to results table
-- Date: 2026-01-26
-- Description: Adds avg_sentence_length, sentence_length_std, special_char_ratio, uppercase_ratio

-- Add new columns (nullable first to avoid breaking existing rows)
ALTER TABLE results ADD COLUMN IF NOT EXISTS avg_sentence_length FLOAT;
ALTER TABLE results ADD COLUMN IF NOT EXISTS sentence_length_std FLOAT;
ALTER TABLE results ADD COLUMN IF NOT EXISTS special_char_ratio FLOAT;
ALTER TABLE results ADD COLUMN IF NOT EXISTS uppercase_ratio FLOAT;

-- Set default values for existing rows (if any)
UPDATE results SET avg_sentence_length = 0.0 WHERE avg_sentence_length IS NULL;
UPDATE results SET sentence_length_std = 0.0 WHERE sentence_length_std IS NULL;
UPDATE results SET special_char_ratio = 0.0 WHERE special_char_ratio IS NULL;
UPDATE results SET uppercase_ratio = 0.0 WHERE uppercase_ratio IS NULL;

-- Make columns non-nullable after setting defaults
ALTER TABLE results ALTER COLUMN avg_sentence_length SET NOT NULL;
ALTER TABLE results ALTER COLUMN sentence_length_std SET NOT NULL;
ALTER TABLE results ALTER COLUMN special_char_ratio SET NOT NULL;
ALTER TABLE results ALTER COLUMN uppercase_ratio SET NOT NULL;

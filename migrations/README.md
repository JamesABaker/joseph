# Database Migrations

## Overview

This directory contains SQL migrations and Python scripts to update the production database schema.

## Current Migrations

### 001: Add New Entropy Features (2026-01-26)

Adds 4 new columns to the `results` table for the 10-feature model:
- `avg_sentence_length` (FLOAT)
- `sentence_length_std` (FLOAT)
- `special_char_ratio` (FLOAT)
- `uppercase_ratio` (FLOAT)

## Running Migrations on Render

### Option 1: Using Render Shell (Recommended)

1. Go to your Render dashboard
2. Navigate to your service (joseph)
3. Click on "Shell" in the left sidebar
4. Run the migration script:
   ```bash
   python migrations/run_migration.py
   ```

### Option 2: Manual SQL (PostgreSQL Console)

1. Go to your Render dashboard
2. Navigate to your PostgreSQL database
3. Click on "Dashboard" → "Connect" → "External Connection"
4. Use the connection string to connect with `psql` or a GUI tool
5. Run the SQL from `001_add_new_entropy_features.sql`

### Option 3: One-time Job on Render

1. In your Render dashboard, create a new "Background Worker" (temporary)
2. Set the start command to: `python migrations/run_migration.py`
3. Deploy and let it run once
4. Delete the worker after successful migration

## Verifying Migration

After running the migration, verify the columns exist:

```sql
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'results'
ORDER BY ordinal_position;
```

You should see the 4 new columns with type `double precision` and `is_nullable = NO`.

## Rollback

If you need to rollback the migration:

```sql
ALTER TABLE results DROP COLUMN IF EXISTS avg_sentence_length;
ALTER TABLE results DROP COLUMN IF EXISTS sentence_length_std;
ALTER TABLE results DROP COLUMN IF EXISTS special_char_ratio;
ALTER TABLE results DROP COLUMN IF EXISTS uppercase_ratio;
```

## Future Migrations

For future schema changes, add new `.sql` files with incrementing numbers:
- `002_description.sql`
- `003_description.sql`
- etc.

Consider setting up Alembic for automated migrations in the future.

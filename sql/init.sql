IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = N'FoodData')
CREATE DATABASE FoodData;
GO

USE FoodData;
GO

-- Создание таблицы, если она еще не существует
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[FoodProducts]') AND type in (N'U'))
CREATE TABLE FoodProducts (
    code BIGINT,
    product_name TEXT,
    energy_kcal_100g FLOAT,
    energy_100g FLOAT,
    fat_100g FLOAT,
    saturated_fat_100g FLOAT,
    trans_fat_100g FLOAT,
    cholesterol_100g FLOAT,
    carbohydrates_100g FLOAT,
    sugars_100g FLOAT,
    fiber_100g FLOAT,
    proteins_100g FLOAT,
    salt_100g FLOAT,
    sodium_100g FLOAT,
    calcium_100g FLOAT,
    iron_100g FLOAT,
    nutrition_score_fr_100g FLOAT,
    categories_en TEXT
);
GO

-- Загрузка данных из CSV файла
BULK INSERT FoodProducts
FROM '/data/openfood.csv'
WITH (
     FIRSTROW = 2,
     FIELDTERMINATOR = ',',
     ROWTERMINATOR = '0x0A',
     TABLOCK
);
GO

-- Добавление столбца prediction после загрузки данных
ALTER TABLE FoodProducts
ADD prediction INT NULL;
GO
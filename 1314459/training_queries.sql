/*
SELECT *
FROM surveys
LIMIT 10 OFFSET 20;

SELECT DISTINCT species_id
FROM surveys;

SELECT DISTINCT day, month, year, species_id, weight*1000
FROM surveys;

SELECT *
FROM surveys
WHERE species_id = 'DM';

SELECT *
FROM surveys
WHERE year >= 2000;

SELECT *
FROM surveys
WHERE (year >= 2000) AND (species_id = 'DM');

SELECT *
FROM surveys
WHERE (year>=2000) AND ((species_id='DM') OR (species_id='DS') OR (species_id='DO'));

SELECT day, month, year, species_id, weight/1000
FROM surveys
WHERE (plot_id = 1) AND (weight>75);

SELECT DISTINCT *
FROM surveys
WHERE (year>=2000) AND (species_id IN ('DM', 'DO', 'DS'));
*/
-- Single Line comment
/*
SELECT year, species_id, weight/1000
FROM surveys
ORDER BY weight DESC;
*/
/*
SELECT *
FROM species
ORDER BY taxa ASC;
*/
-- SELECT year, species_id, weight/1000
-- FROM surveys
-- ORDER BY weight DESC;

-- SELECT day, month, year, species_id, ROUND(weight/1000,2)
-- FROM surveys
-- WHERE (year = 1999)
-- ORDER BY species_id ASC;

-- SELECT COUNT(*)
-- FROM surveys
-- WHERE year>1999;

-- SELECT COUNT(*), SUM(ROUND(weight/1000,2)), MAX(weight/1000), MIN(weight/1000), AVG(weight/1000)
-- FROM surveys
-- WHERE weight BETWEEN 5 AND 10;

-- SELECT species_id, COUNT(*)
-- FROM surveys
-- GROUP BY species_id;

-- Count of individuals per year
-- SELECT year, COUNT(*)
-- FROM surveys
-- GROUP BY year;

-- Count per year per species
-- SELECT species_id, year, COUNT(*), AVG(weight)
-- FROM surveys
-- GROUP BY year, species_id

-- SELECT MAX(year) AS last_survey_year
-- FROM surveys;

-- SELECT species_id, COUNT(species_id) AS occurrencies
-- FROM surveys
-- GROUP BY species_id
-- HAVING occurrencies > 10;

-- SELECT species.species,taxa, COUNT(species.species) AS n
-- FROM species
-- GROUP BY taxa
-- HAVING n>10;
-- 
-- SELECT *
-- FROM surveys
-- WHERE (year = 2000) AND (month BETWEEN 4 AND 10);

-- SELECT *
-- FROM surveys
-- LEFT OUTER JOIN species
-- USING (species_id);

SELECT species.genus, species.species, surveys.weight
FROM species
JOIN surveys
USING (species_id);
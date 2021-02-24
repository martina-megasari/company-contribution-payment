WITH MONTHLY_CONTRIBUTION AS
(WITH DT_QUARTERLY AS (
    WITH Q AS (
        SELECT CONTRIBUTION_ID
             , ENDS_ON                       AS ENDS_ON_1
             , DATEADD('month', -1, ENDS_ON) as ENDS_ON_2
             , DATEADD('month', -2, ENDS_ON) as ENDS_ON_3
        FROM FACT_CONTRIBUTIONS FC
        WHERE 1 = 1
          AND PERIOD_TYPE IN ('Quarterly')
          AND IS_REFUNDED = FALSE
          AND PAYMENT_HAS_ISSUE = FALSE
        ORDER BY CONTRIBUTION_ID
    )
    select CONTRIBUTION_ID, YEAR(DT) AS DT_Y, MONTH(DT) AS DT_M
    from Q
        unpivot (DT for month in (ENDS_ON_1, ENDS_ON_2, ENDS_ON_3))
    order by CONTRIBUTION_ID
),
DT_BIANNUALLY AS (
    WITH Q AS (
        SELECT CONTRIBUTION_ID
             , ENDS_ON                       AS ENDS_ON_1
             , DATEADD('month', -1, ENDS_ON) as ENDS_ON_2
             , DATEADD('month', -2, ENDS_ON) as ENDS_ON_3
             , DATEADD('month', -3, ENDS_ON) as ENDS_ON_4
             , DATEADD('month', -4, ENDS_ON) as ENDS_ON_5
             , DATEADD('month', -5, ENDS_ON) as ENDS_ON_6
        FROM FACT_CONTRIBUTIONS FC
        WHERE 1 = 1
          AND PERIOD_TYPE IN ('BiAnnually')
          AND IS_REFUNDED = FALSE
          AND PAYMENT_HAS_ISSUE = FALSE
        ORDER BY CONTRIBUTION_ID
    )
    select CONTRIBUTION_ID, YEAR(DT) AS DT_Y, MONTH(DT) AS DT_M
    from Q
        unpivot (DT for month in (ENDS_ON_1, ENDS_ON_2, ENDS_ON_3, ENDS_ON_4, ENDS_ON_5, ENDS_ON_6))
    order by CONTRIBUTION_ID, DT_Y, DT_M
),
DT_ANNUALLY AS (
    WITH Q AS (
        SELECT CONTRIBUTION_ID
             , ENDS_ON                       AS ENDS_ON_1
             , DATEADD('month', -1, ENDS_ON) as ENDS_ON_2
             , DATEADD('month', -2, ENDS_ON) as ENDS_ON_3
             , DATEADD('month', -3, ENDS_ON) as ENDS_ON_4
             , DATEADD('month', -4, ENDS_ON) as ENDS_ON_5
             , DATEADD('month', -5, ENDS_ON) as ENDS_ON_6
             , DATEADD('month', -6, ENDS_ON) as ENDS_ON_7
             , DATEADD('month', -7, ENDS_ON) as ENDS_ON_8
             , DATEADD('month', -8, ENDS_ON) as ENDS_ON_9
             , DATEADD('month', -9, ENDS_ON) as ENDS_ON_10
             , DATEADD('month', -10, ENDS_ON) as ENDS_ON_11
             , DATEADD('month', -11, ENDS_ON) as ENDS_ON_12
        FROM FACT_CONTRIBUTIONS FC
        WHERE 1 = 1
          AND PERIOD_TYPE IN ('Annually')
          AND IS_REFUNDED = FALSE
          AND PAYMENT_HAS_ISSUE = FALSE
        ORDER BY CONTRIBUTION_ID
    )
    select CONTRIBUTION_ID, YEAR(DT) AS DT_Y, MONTH(DT) AS DT_M
    from Q
        unpivot (DT for month in (ENDS_ON_1, ENDS_ON_2, ENDS_ON_3, ENDS_ON_4, ENDS_ON_5, ENDS_ON_6,
            ENDS_ON_7, ENDS_ON_8, ENDS_ON_9, ENDS_ON_10, ENDS_ON_11, ENDS_ON_12))
    order by CONTRIBUTION_ID, DT_Y, DT_M
)
SELECT
     COMPANY_ID
     , YEAR(ENDS_ON) Y
     , MONTH(ENDS_ON) M
     , SUM(COMPANY_AMOUNT_PENNIES) AS COMPANY_AMOUNT_PENNIES
FROM FACT_CONTRIBUTIONS
WHERE 1 = 1
AND PERIOD_TYPE IN ('Weekly','FourWeekly','Fortnightly')
AND IS_REFUNDED = FALSE
AND PAYMENT_HAS_ISSUE = FALSE
AND CONTRIBUTION_STATE IN (
     'released',
     'created',
     'payment_pending',
     'investing',
     'invested',
     'paid')
GROUP BY COMPANY_ID, YEAR(ENDS_ON), MONTH(ENDS_ON)

UNION

SELECT
       FC.COMPANY_ID
     , YEAR(ENDS_ON) AS Y
     , MONTH(ENDS_ON) AS M
     , FC.COMPANY_AMOUNT_PENNIES
FROM FACT_CONTRIBUTIONS FC
WHERE 1 = 1
AND CONTRIBUTION_STATE IN (
     'released',
     'created',
     'payment_pending',
     'investing',
     'invested',
     'paid')
AND PERIOD_TYPE IN ('Monthly')
AND IS_REFUNDED = FALSE
AND PAYMENT_HAS_ISSUE = FALSE

UNION

SELECT
       FC.COMPANY_ID
     , DT_Y AS Y
     , DT_M AS M
     , FC.COMPANY_AMOUNT_PENNIES/(12/PERIOD_TYPE_TO_NUMBER(FC.PERIOD_TYPE)) as COMPANY_AMOUNT_PENNIES
FROM FACT_CONTRIBUTIONS FC
JOIN DT_QUARTERLY ON DT_QUARTERLY.CONTRIBUTION_ID = FC.CONTRIBUTION_ID
WHERE 1 = 1
AND CONTRIBUTION_STATE IN (
     'released',
     'created',
     'payment_pending',
     'investing',
     'invested',
     'paid')
AND PERIOD_TYPE IN ('Quarterly')
AND IS_REFUNDED = FALSE
AND PAYMENT_HAS_ISSUE = FALSE

UNION

SELECT
       FC.COMPANY_ID
     , DT_Y AS Y
     , DT_M AS M
     , FC.COMPANY_AMOUNT_PENNIES/(12/PERIOD_TYPE_TO_NUMBER(FC.PERIOD_TYPE)) as COMPANY_AMOUNT_PENNIES
FROM FACT_CONTRIBUTIONS FC
JOIN DT_BIANNUALLY ON DT_BIANNUALLY.CONTRIBUTION_ID = FC.CONTRIBUTION_ID
WHERE 1 = 1
AND CONTRIBUTION_STATE IN (
     'released',
     'created',
     'payment_pending',
     'investing',
     'invested',
     'paid')
AND PERIOD_TYPE IN ('BiAnnually')
AND IS_REFUNDED = FALSE
AND PAYMENT_HAS_ISSUE = FALSE

UNION

SELECT
       FC.COMPANY_ID
     , DT_Y AS Y
     , DT_M AS M
     , FC.COMPANY_AMOUNT_PENNIES/(12/PERIOD_TYPE_TO_NUMBER(FC.PERIOD_TYPE)) as COMPANY_AMOUNT_PENNIES
FROM FACT_CONTRIBUTIONS FC
JOIN DT_ANNUALLY ON DT_ANNUALLY.CONTRIBUTION_ID = FC.CONTRIBUTION_ID
WHERE 1 = 1
AND CONTRIBUTION_STATE IN (
     'released',
     'created',
     'payment_pending',
     'investing',
     'invested',
     'paid')
AND PERIOD_TYPE IN ('Annually')
AND IS_REFUNDED = FALSE
AND PAYMENT_HAS_ISSUE = FALSE
    )
SELECT
       MC.COMPANY_ID
     , MC.Y
     , MC.M
     , CAST(ROUND(SUM(MC.COMPANY_AMOUNT_PENNIES)) AS INT) AS COMPANY_AMOUNT_PENNIES
FROM MONTHLY_CONTRIBUTION MC
GROUP BY MC.COMPANY_ID, MC.Y, MC.M
ORDER BY MC.COMPANY_ID, MC.Y, MC.M
;
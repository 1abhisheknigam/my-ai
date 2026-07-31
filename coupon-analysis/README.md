# Customer Coupon Acceptance Analysis

## Overview

An analytical exploration of driver behaviors and situational factors that influence whether a customer accepts a mobile coupon delivered while driving. This study utilizes data visualization and conditional probability distributions on the given dataset to attempt to answer this primary question: **which factors drive the acceptance rate of a coupon**?

To do this, this project utilizes **Python**, **Pandas**, and **NumPy** for data engineering alongside **Matplotlib** and **Seaborn** for <u>exploratory data analysis (EDA)</u>. this study identifies how target demographics, environmental constraints (weather, time of day), and passenger contexts compound to predict coupon validation. The final findings serve not only as providing context for coupon acceptance, but also as recommendations to drive further areas of acceptance growth.

## Dataset

- **Source:** UCI Machine Learning Repository (Amazon Mechanical Turk survey)
- **Size:** 12,684 observations, 26 features:
  
  ```
    #   Column                Comment
    ---  ------               --------------  ----- 
    0   destination           headed to "home", "office", "other"
    1   passanger  [sic]      with "kid(s)", "partner", "friend(s)", "Alone"
    2   weather               "sunny", "rainy", "snowy"
    3   temperature           fixed values representing "cold", "warm", "hot"
    4   time                  fixed values represetnting time of day
    5   coupon                category of coupon e.g. "bar", "CoffeeHouse"
    6   expiration            time till expiry ranging from `2h` to `1d`
    7   gender                "Male" or "Female"
    8   age                   fixed values representing stages of life
    9   maritalStatus         type of partner, single or widowed
    10  has_children          1,2 or more kids
    11  education             fixed values representing highest education
    12  occupation            various industries
    13  income                various income ranges
    14  car                   type of vehicle when coupon was received
    15  Bar                   fixed values representing visit frequency
    16  CoffeeHouse           fixed values representing visit frequency
    17  CarryAway             fixed values representing visit frequency
    18  RestaurantLessThan20  fixed values representing visit frequency
    19  Restaurant20To50      fixed values representing visit frequency
    20  toCoupon_GEQ5min      single bit representing time to coupon location
    21  toCoupon_GEQ15min     single bit representing time to coupon location
    22  toCoupon_GEQ25min     single bit representing time to coupon location
    23  direction_same        single bit representing coupon direction along the way
    24  direction_opp         single bit representing coupon direction opposite the way
    25  Y                     Target Value: Single bit representing coupon was accepted
  ```

## Data Cleaning & Preprocessing

An initial investigation of the 12,684 records revealed several columns with missing (`NaN`) values:

*   **`car` (99.15% missing):** Due to the near-total absence of data (12,576 missing rows), this column is stripped from systemic analysis as it lacks statistical significance.
*   **Behavioral Frequency Columns (`Bar`, `CoffeeHouse`, `CarryAway`, `RestaurantLessThan20`, `Restaurant20To50`):** Missing values ranging from 0.84% to 1.71% are addressed programmatically. Since these represent ordinal habit ranges (e.g., "never", "less1", "1~3"), missing rows are either imputed using the mode distribution or isolated to protect statistical integrity.

---

## Key Findings by Core Problems

### 1. Overall Acceptance Rate

- **56.93%** of all coupons were accepted
- Acceptance rate varies between different coupon categories significantly: here is the each category's share of the accepted coupons:

![A Pie Chart showing the breakdown of coupon acceptance by category](outputs/coupon_acceptance_pie_chart.png)

- Acceptance rate was heavily influecnced by visit frequency across all categories - regular visitors were more likely to use a coupon for their coupon category of their destination

### 2. Bar Coupons Analysis

- As seen above, bar coupons had the **lowest** acceptance rate.
- **Habitual Drivers vs. Others:** Drivers who visit bars at least once a month display a significantly higher coupon acceptance rate compared to those who go less frequently or never.
- **The Age & Social Dynamic:** Younger cohorts (under age 30) exhibit an increased probability of accepting bar coupons, particularly when driving with friends as passengers compared to driving solo.
- **Occupational Flexibility:** Drivers with flexible occupations or non-traditional corporate structures show higher conversion rates for spontaneous detours.

### 3. Coffee House Coupons Analysis

- As seen above, Coffee House coupons have the **highest** acceptance rate.
- **Temporal Urgency:** Coffee house coupon acceptance heavily peaks during morning commute hours (7 AM – 9 AM).
- **Destination Constraints:** Drivers on their way to "Work" or "Home" decline coffee house coupons at higher rates than drivers indicating "No Urgent Destination".
- **Passenger Influence:** Driving with a partner or friends boosts coffee house coupon validation rates, indicating that caffeine stops are frequently treated as shared social activities.


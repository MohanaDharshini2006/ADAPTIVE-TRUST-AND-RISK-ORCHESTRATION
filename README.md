# ATRO Billing Agent – Data Preprocessing

## Overview

This notebook focuses on collecting, cleaning, filtering, and preparing datasets for the Billing Agent in the Adaptive Trust & Risk Orchestrator (ATRO) project.

The prepared datasets support billing-related customer queries, payment and refund analysis, invoice lookup, and risk assessment.

## Objectives

* Clean and preprocess customer support data.
* Extract billing-related customer questions and answers.
* Identify billing-related fashion product reviews.
* Process retail transactions and generate invoice summaries.
* Prepare structured CSV files for model training and evaluation.

## Datasets Used

### 1. Customer Support Dataset

Used to extract billing-related questions and reference answers.

**Preprocessing steps:**

* Inspect missing values and duplicate records.
* Remove unrelated support categories.
* Fix text encoding issues and normalize whitespace.
* Filter billing-related questions using keywords.
* Assign billing intent categories.
* Remove duplicate questions.
* Generate document IDs and standardize column names.

### 2. Amazon Fashion Reviews

Used to identify customer reviews related to billing, refunds, payments, charges, and discounts.

**Preprocessing steps:**

* Clean review text and summaries.
* Remove short reviews and duplicate reviews.
* Identify reviews with ratings of 3 or below.
* Categorize reviews using billing-related keyword patterns.
* Limit the number of samples per category to improve balance.
* Standardize the dataset structure.

### 3. Online Retail Dataset

Used to prepare transaction records and invoice-level summaries.

**Preprocessing steps:**

* Inspect missing values, duplicates, and invalid transaction values.
* Remove duplicate records.
* Normalize invoice numbers, stock codes, and product descriptions.
* Remove records with negative unit prices while retaining negative quantities.
* Calculate transaction amounts.
* Identify sales, cancellations, and returns or adjustments.
* Generate invoice-level summaries.

## Output Files

The notebook generates the following CSV files:

| File                          | Purpose                                                                      |
| ----------------------------- | ---------------------------------------------------------------------------- |
| `billing_support_qa.csv`      | Billing-related customer questions, answers, and intent labels               |
| `billing_amazon_reviews.csv`  | Filtered billing-related Amazon Fashion reviews                              |
| `billing_rag_corpus.csv`      | Combined customer support and review data for retrieval-augmented generation |
| `billing_transactions.csv`    | Cleaned retail transaction records                                           |
| `billing_invoice_summary.csv` | Invoice-level transaction summaries                                          |

## Technologies Used

* Python
* Pandas
* NumPy
* Regular Expressions
* ftfy
* Google Colab

## Workflow

1. Upload and extract the datasets.
2. Inspect and clean the data.
3. Extract billing-related records.
4. Categorize customer queries and reviews.
5. Process retail transactions.
6. Generate invoice summaries.
7. Export the prepared datasets as CSV files.

## How to Run

1. Open the notebook in Google Colab.
2. Upload the required dataset ZIP file as `data.zip`.
3. Run the notebook cells sequentially.
4. Verify the generated CSV files.
5. Use the output files as inputs to the ATRO Billing Agent training notebook.

## Role in ATRO

This notebook prepares the structured data required by the Billing Agent. The resulting datasets support customer query classification, evidence retrieval, invoice lookup, and downstream trust and risk evaluation.

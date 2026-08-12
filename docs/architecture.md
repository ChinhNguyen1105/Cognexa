# Cognexa V0.1 Architecture

## Overview

Cognexa V0.1 is a document processing application.

Its main responsibility is to load a document, parse its content,
clean the text, and return a processed Document object.

## Pipeline

```text
Input File
    |
    v
+-----------+
|  Ingestion|
+-----------+
    |
    v
  Document
    |
    v
+-----------+
|  Parsing  |
+-----------+
    |
    v
 Parsed Text
    |
    v
+-----------+
|  Cleaning |
+-----------+
    |
    v
Processed Text
    |
    v
Document
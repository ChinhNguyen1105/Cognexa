                         COGNEXA V0.1
                              │
                              ▼
                   scripts/process_document
                              │
                              ▼
                    services.document_processor
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                ingestion           processing
                    │                   │
             ┌──────┴──────┐      ┌────┴─────┐
             ▼             ▼      ▼          ▼
          loader        reader  parser     cleaner
             │
             ▼
          Document
             │
             └───────────────► Processed Document
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


v0.2 - typer 

                    Cognexa V0.2

                 CLI Application
                       │
                       ▼
              cognexa process FILE
                       │
                       ▼
              DocumentProcessor
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Ingestion             Processing
        load_document()       process_content()
             │                   │
             └─────────┬─────────┘
                       ▼
                   Document


cognexa- v0.3
                         COGNEXA V0.3
                              │
                              ▼
                         CLI: cognexa
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
            process          train         classify
               │              │              │
               ▼              ▼              ▼
       document_processor   dataset      classifier
               │              │              │
        ┌──────┴──────┐       │              │
        ▼             ▼       ▼              │
    ingestion     processing  CSV            │
        │             │       │              │
   ┌────┴────┐    ┌───┴───┐   ▼              │
   ▼         ▼    ▼       ▼  texts + labels   │
 loader    reader parser cleaner              │
   │             │       │                    │
   └─────────────┴───────┘                    │
                 │                            │
                 ▼                            │
             Document                         │
                 │                            │
                 ▼                            │
          Processed Document                  │
                                              │
                         ┌────────────────────┘
                         │
                         ▼
                  TextFeatureExtractor
                         │
                         ▼
                       TF-IDF
                         │
                         ▼
                  DocumentClassifier
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
            train                 predict
              │                     │
              ▼                     ▼
        trained model          Prediction
              │
              ▼
       cognexa_model.pkl
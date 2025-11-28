## Repository for DSAI4401 - Applied Data Mining project

### Team members 
- Djihane Mahraz 
- Amina Baziz

- three things needed for the pipeline: the embedding model, the anomaly detector, and the few-shot recognizer. The embedding model (ResNet18) is stored in the models/classifier folder and is used only to turn YOLO crops into 256-dimensional feature vectors. The anomaly detector is in models/anomaly/lof_scorer.pkl, and it tells whether a crop is normal trash or an unusual item. If the object is unusual, the few-shot recognizer in models/fewshot/prototypes.pkl tries to identify it as clothes, electronics, food, or toys, or returns “unknown.” Person A doesn’t need any datasets or training code—just these files and the functions that load them. Together, these components let the detection pipeline classify normal trash, detect unknown objects, and label rare categories.
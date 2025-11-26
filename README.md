# RAG over Images
This repository contains RAG based system to query and find necessary objects from the image


System key requirements:
A user will upload multiple images (put a reasonable restriction on number of images)
- The user will ask if a particular object (imagine a car, a sign-board, a machine) is
present in the image.
- The user will describe the object in few words in the query (for example: signage of a
retail shop, a black sedan car)
- The system will try to locate it , answer the presence/absence question and also
highlight the object in the image


Considerations:
1. Assume one single object is queried at a time.
2. Use locally hosted LLM/VLM
3. Use memory datra stores as needed. 
4. Query side can be demonstrated using LLM APIs. (but ingestion is prefered in local/open LLM)


Dataset: 
pascal-voc-2012-dataset
https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset
# Stylizer
Python server on FastAPI which serves as "stylizer" of the images. Under the hood uses model from tensorflow-hub.

Related to: https://arxiv.org/pdf/1705.06830.pdf.

For examples refer to test_image.ipynb.

# To deploy on gcloud
1) Create a project on gcloud.
2) Install gcloud and authorise with your account.
3) Set project_id.
3) In the folder of the project execute the following:
<pre>gcloud app deploy app.yaml -v v1</pre>

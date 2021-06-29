from django.shortcuts import render
from .apps import PredictorConfig
from django.http import JsonResponse
from rest_framework.views import APIView

from django.template import RequestContext
from django.http import HttpResponseRedirect
# from django.core.urlresolvers import reverse

from .models import Document
from .forms import DocumentForm
# from .predict import classifier

def index(request):
    form = DocumentForm() # A empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()
    return render(request, "upload.html", {'documents': documents,'form': form})

def call_model(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            doc = request.FILES['docfile']
            newdoc = Document(docfile = request.FILES['docfile'])
            newdoc.save()

            imagepath =  Document.objects.get(id=newdoc.id)

            # print(doc)
            import tensorflow as tf
            from keras.models import load_model

            classifier = load_model('predictor/models/cnn_healthy_unhealthy_model.h5')
            # classifier = tf.lite.TFLiteConverter.from_keras_model('models/cnn_healthy_unhealthy_model.h5')

            import numpy as np
            from keras.preprocessing import image
            test_image = image.load_img(imagepath.docfile.path, target_size = (64, 64))

            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = classifier.predict(test_image)

            if result[0][0] == 1:
                prediction = 'unhealthy'
            else:
                prediction = 'healthy'

            print("\nPrediction: " + prediction)
            # Redirect to the document list after POST
            # return HttpResponseRedirect(('/classify'))
            return render(request, "result.html", {'prediction': prediction})
    else:
        form = DocumentForm() # A empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()

    # Render list page with the documents and the form
    return render(request, "result.html")
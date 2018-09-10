import sys,os
lib_path = os.path.join(os.environ['URESNETDIR'],'lib')
if not lib_path in sys.path:
    print lib_path
    sys.path.insert(0,lib_path)

import ssnet_trainval as api
import tensorflow as tf
import sys
import numpy as np
#
# What does this script do?
# 0) Run u-resnet inference: you need to give a config file in an argument
# 1) Compute class-wise accuracy
# 2) Store results in CSV file
# ... there is a separate example script, ana_csv.py, to analyze CSV file created in step 2)

# instantiate ssnet_trainval and configure
img_height=128
img_width=96
t = api.ssnet_trainval()
for argv in sys.argv:
    if argv.endswith('.cfg'): t.override_config(argv)

confusionMatrix = np.zeros([3, 3])
# initialize
t.initialize()

# get number of classes
num_class = t.num_class()


#
# run interactive analysis
#
ITERATIONS=200
for iteration in np.arange(ITERATIONS):

    # call "ana_step" API function: this runs network inference and returns results
    res = t.ana_step()
    # The results, "res", is a dictionary (try running ana_example.py to see by yourself)
    # Contents are:
    # 0) res['entries'    ] = 1D numpy array that holds TTree entry numbers analyzed in this batch
    # 1) res['acc_all'    ] = an average accuracy across all pixels over all images in a batch
    # 2) res['acc_nonzero'] = an average accuracy across all non-zero pixels over all images in a batch
    # 3) res['input'      ] = a batch of input images (numpy array), dimension [Batch,Row,Col,Channels]
    # 4) res['label'      ] = a batch of label images (numpy array), dimension [Batch,Row,Col,Channels]
    # 5) res['softmax'    ] = a batch of softmax images (numpy array), dimension [Batch,Row,Col,Channels]

    entries_batch = res['entries']
    softmax_batch = res['softmax']
    image_batch   = res['input'  ]
    label_batch   = res['label'  ]
    
    # Note: Next, we loop over images in this batch and compute analysis variables.
    
    # Loop over each entry = image in this batch
    #print("softmax_batch", len(softmax_batch))
    #print("softmax_batch", np.arange(len(softmax_batch)))
    for index in np.arange(len(softmax_batch)):

        entry   = entries_batch[index]
        softmax = softmax_batch[index]
        label   = label_batch[index]

        # Let's squeeze the label: this changes dimension from (row,col,1) to (row,col)
        label = np.squeeze(label)
        #print("label",label)
        # Compute acc_all
        prediction = np.argmax(softmax,axis=-1)
        print "prediction shape=", prediction.shape
        print "label shape=", label.shape
        for h in xrange(img_height):
            for w in xrange(img_width):
              #  print "prediction inside loop", prediction[h][w]
              #  print "label inside loop", label[h][w].astype(np.int32)
                confusionMatrix[prediction[h][w], label[h][w].astype(np.int32)] += 1 
np.save("semsegconf", confusionMatrix)
t.reset()

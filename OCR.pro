QT += core
QT -= gui

CONFIG += c++11

TARGET = OCR
CONFIG += console
CONFIG -= app_bundle

#pkg-config '--cflags --libs opencv'
CONFIG += link_pkgconfig
PKGCONFIG += opencv

LIBS += -L/usr/local/lib/ -lopencv_dnn -lopencv_xfeatures2d

TEMPLATE = app

SOURCES += main.cpp \
    farsi_ocr/cvBlobsLib/blob.cpp \
    farsi_ocr/cvBlobsLib/BlobContour.cpp \
    farsi_ocr/cvBlobsLib/BlobOperators.cpp \
    farsi_ocr/cvBlobsLib/BlobResult.cpp \
    farsi_ocr/cvBlobsLib/ComponentLabeling.cpp \
    farsi_ocr/Character.cpp \
    farsi_ocr/DistanceProfile.cpp \
    farsi_ocr/functions.cpp \
    farsi_ocr/lettetRecognizer.cpp

HEADERS += \
    farsi_ocr/cvBlobsLib/blob.h \
    farsi_ocr/cvBlobsLib/BlobContour.h \
    farsi_ocr/cvBlobsLib/BlobLibraryConfiguration.h \
    farsi_ocr/cvBlobsLib/BlobOperators.h \
    farsi_ocr/cvBlobsLib/BlobResult.h \
    farsi_ocr/cvBlobsLib/ComponentLabeling.h \
    farsi_ocr/Character.h \
    farsi_ocr/DistanceProfile.h \
    farsi_ocr/functions.h \
    tiny-cnn/activation_function.h \
    tiny-cnn/average_pooling_layer.h \
    tiny-cnn/cifar10_parser.h \
    tiny-cnn/config.h \
    tiny-cnn/convolutional_layer.h \
    tiny-cnn/deform.h \
    tiny-cnn/display.h \
    tiny-cnn/dropout.h \
    tiny-cnn/fully_connected_dropout_layer.h \
    tiny-cnn/fully_connected_layer.h \
    tiny-cnn/image.h \
    tiny-cnn/input_layer.h \
    tiny-cnn/layer.h \
    tiny-cnn/layers.h \
    tiny-cnn/loss_function.h \
    tiny-cnn/max_pooling_layer.h \
    tiny-cnn/mnist_parser.h \
    tiny-cnn/network.h \
    tiny-cnn/optimizer.h \
    tiny-cnn/partial_connected_layer.h \
    tiny-cnn/product.h \
    tiny-cnn/tiny_cnn.h \
    tiny-cnn/util.h \
    tiny-cnn/weight_init.h \
    farsi_ocr/letterRecognizer.h

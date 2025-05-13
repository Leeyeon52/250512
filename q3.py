# CNN 생성에 필요한 Keras 라이브러리, 패키지
import os

import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


def create_classifier():
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(32, (3, 3), activation="relu"))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=128, activation="relu"))
    classifier.add(Dense(units=1, activation="sigmoid"))
    classifier.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return classifier


def generate_dataset(rescale_ratio, horizontal_flip, train_dir, test_dir):
    train_datagen = ImageDataGenerator(
        rescale=rescale_ratio, horizontal_flip=horizontal_flip
    )

    test_datagen = ImageDataGenerator(rescale=rescale_ratio)

    training_set = train_datagen.flow_from_directory(
        train_dir, target_size=(64, 64), batch_size=2, class_mode="binary"
    )

    test_set = test_datagen.flow_from_directory(
        test_dir, target_size=(64, 64), batch_size=2, class_mode="binary"
    )

    return train_datagen, test_datagen, training_set, test_set


def main():
    classifier = create_classifier()

    """지시사항 1. `rescale_ratio`의 값을 설정하세요. """
    # 픽셀 값을 0-1 범위로 조정하기 위해 255로 나눕니다.
    rescale_ratio = 1.0 / 255

    """지시사항 2. `horizontal_flip`을 사용하도록 설정하세요. """
    # horizontal_flip을 True로 설정하여 이미지 좌우 반전을 활성화합니다.
    horizontal_flip = True

    """지시사항 3. 전처리된 훈련 및 테스트 데이터 전처리기 및 데이터셋을 생성하세요. """
    # 훈련 및 테스트 데이터셋의 경로를 지정합니다.
    train_dir = r"D:\Medical-practice-main\q3\dataset\training_set"
    test_dir = r"D:\Medical-practice-main\q3\dataset\test_set"
    # generate_dataset 함수를 호출하여 ImageDataGenerator 객체와 데이터셋을 생성합니다.
    train_datagen, test_datagen, training_set, test_set = generate_dataset(
        rescale_ratio, horizontal_flip, train_dir, test_dir
    )

    classifier.fit_generator(
        training_set,
        steps_per_epoch=10,
        epochs=10,
        validation_data=test_set,
        validation_steps=10,
    )

    output = classifier.predict_generator(test_set, steps=5)
    print(test_set.class_indices)

    return rescale_ratio, horizontal_flip, train_datagen, test_datagen, training_set, test_set, classifier, output


if __name__ == "__main__":
    main()
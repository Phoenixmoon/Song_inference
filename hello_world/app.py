import json
from PIL import Image
import numpy as np
import onnxruntime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Dict, Tuple, Any, Optional, Union
import os
import boto3


def get_bucket_name_and_key(s3_uri):
    components = s3_uri.split("/")
    bucket_name = components[2]

    # The key is everything after the bucket name, joined by "/"
    key = "/".join(components[3:])

    # Get the filename from the key
    filename = key.split("/")[-1]

    return bucket_name, key, filename


def onnx_prediction(
        test_ims: List[str],
        model_path: str,
        output_test_dir: Path = None,
        batch_size: int = 16,
        size: Tuple[int, int] = (128, 128),
) -> Union[str, List[str]]:

    tmp_dir = TemporaryDirectory()
    if not output_test_dir:
        output_test_dir = Path(tmp_dir.name)

    batch_images = []
    for i in test_ims:
        image = Image.open(str(i)).resize(size, resample=Image.NEAREST)
        im_array = np.array(image).reshape(size[0], size[1], 1) / 255
        batch_images.append(im_array)

    batch_images = np.array(batch_images)
    preprocessed_images = batch_images.astype(np.float32)

    batch_votes = []

    session = onnxruntime.InferenceSession(model_path)

    for batch_index in range(0, len(test_ims), batch_size):
        batch = preprocessed_images[batch_index:batch_index + batch_size]

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        predictions = session.run([output_name], {input_name: batch})

        predictions = predictions[0]

        # print(predictions)
        classical_probability = 0
        for prediction in predictions:
            classical_probability += prediction[2]

        rock_probability = 0
        for prediction in predictions:
            rock_probability += prediction[0]

        blues_probability = 0
        for prediction in predictions:
            blues_probability += prediction[1]

        probabilities = {"Classical": classical_probability, "Alternative/Rock": rock_probability,
                         "Blues/Jazz": blues_probability}

        final_vote = max(probabilities, key=probabilities.get)

        batch_votes.append(final_vote)

        tmp_dir.cleanup()

        if len(batch_votes) == 1:
            return final_vote

    return batch_votes


def lambda_handler(event, context):
    """Sample pure Lambda function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """

    querystring = event.get('queryStringParameters', event)
    im_list = querystring.get('im_s3_list')
    model_s3_path = querystring.get('model_s3_path', 's3://musicclassifier/Model/10_sec_CNN.onnx')

    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)

        # download the images
        s3 = boto3.client('s3')
        # s3 = boto3.resource('s3')
        bucket_name = 'musicclassifierspectrograms'

        for s3_key in im_list:
            try:
                s3.download_file(bucket_name, s3_key, s3_key)
                # s3.meta.client.download_file(bucket_name, s3_key, s3_key)
            except Exception as e:
                print(f"Error downloading image {s3_key}: {e}")

        ### Temporary:
        model_bucket, model_key, model_filename = get_bucket_name_and_key(model_s3_path)
        try:
            s3.download_file(model_bucket, model_key, model_filename)
        except Exception as e:
            print(f"Couldn't download model: {e}")

        prediction = onnx_prediction(im_list, model_filename)

        print(prediction)


        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",  # Change this to your specific allowed origins
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
            },
            "body": json.dumps({
                "Prediction": prediction
            }),
        }


if __name__ == "__main__":
    im_list = [f'test{i}.png' for i in range(16)]
    lambda_handler({"queryStringParameters": {"im_s3_list": im_list}}, None)
    print(im_list)
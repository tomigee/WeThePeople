import boto3
import botocore
import os
import time
import threading


def upload_to_s3(client, bucket_name, s3_key_prefix, dirs):
    for item in dirs:
        if os.path.isdir(item):
            entries = [os.path.join(item, entry) for entry in os.listdir(item)]
            upload_to_s3(client, bucket_name, s3_key_prefix, entries)
        else:
            s3_filepath = s3_key_prefix + item
            try:
                s3_file_exists(client, bucket_name, s3_filepath)
            except botocore.exceptions.ClientError:
                client.upload_file(item, bucket_name, s3_filepath)


def s3_file_exists(client, bucket: str, key: str):
    response = client.head_object(Bucket=bucket, Key=key)
    return response


def upload(local_dir, bucket, key_prefix, n_threads):

    # AWS Setup, User settable
    session = boto3.Session()
    s3 = session.client('s3')

    data_entries = [os.path.join(local_dir, item)
                    for item in sorted(os.listdir(local_dir))]
    threads = []
    start_index = 0
    t0 = time.time()
    for i in range(n_threads):
        end_index = round((i+1) * len(data_entries)/n_threads)
        thread_shard = data_entries[start_index: end_index]
        thread = threading.Thread(
            target=upload_to_s3, args=(s3, bucket, key_prefix, thread_shard)
        )
        threads.append(thread)
        thread.start()
        start_index = end_index

    for thread in threads:
        thread.join()
    t1 = time.time()

    print(f"Done, Total: {t1-t0} seconds")

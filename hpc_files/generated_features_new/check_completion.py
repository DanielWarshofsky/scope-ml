import subprocess
import yaml


if __name__ == '__main__':
    config_path = '../../config.yaml'
    with open(config_path, mode='r') as file:
        config_data = yaml.load(file)
    fields_to_check = config_data['feature_generation']['fields_to_run']
    for field in fields_to_check:
        num = subprocess.getoutput(f'ls field_{field}/*.parquet -1 | wc -l')
        if int(num) != 64:
            print(f'{field} does not have 64 feature files it has {num}')
        else:
            print(f'{field} is done')

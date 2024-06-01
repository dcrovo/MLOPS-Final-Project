import requests
from typing import Tuple, Any

def get_data_from_api(api_url: str = "http://10.43.101.149:80", group_number: int = 6) -> Tuple[Any, int]:
    """
    Fetch data from the API for the specified group number.

    Parameters:
    api_url (str): The base URL of the API. Default is "http://10.43.101.149:80".
    group_number (int): The group number to fetch data for. Default is 6.

    Returns:
    Tuple[Any, int]: A tuple containing the data and the batch number.

    Raises:
    Exception: If there is an error with the API request.
    """
    req = f"{api_url}/data?group_number={group_number}"
    
    response = requests.get(req)
    if response.status_code == 200: 
        response_json = response.json()
        data = response_json.get('data')
        batch_number = response_json.get('batch_number')
        return data, batch_number

    if response.status_code == 422:
        raise Exception("Validation Error")
    else: 
        raise Exception(f"Error with status code: {response.status_code}")

def restart_data_generation(api_url: str = "http://10.43.101.149:80", group_number: int = 6) -> None:
    """
    Restart the data generation process for the specified group number.

    Parameters:
    api_url (str): The base URL of the API.
    group_number (int): The group number for which to restart data generation.

    Raises:
    Exception: If there is an error with the API request.
    """
    req = f"{api_url}/restart_data_generation?group_number={group_number}"
    
    response = requests.get(req)
    if response.status_code == 200:
        print("Data restarted successfully")
    elif response.status_code == 422:
        print("Validation error")
    else: 
        raise Exception(f"Error with status code: {response.status_code}")
from __future__ import annotations
import argparse
import base64
from collections import defaultdict
import itertools
import io
import os
import sys
from typing import (
    Any,
    Iterator,
    NoReturn,
    Optional,
)
import uuid

from PIL import Image
from google.auth.transport.requests import Request
import googleapiclient.discovery
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import vobject


THIS_DIR_PATH = os.path.dirname(os.path.normpath(__file__))

INPUT_OUTPUT_PATH_HELP = '''
input_path and output_path can be a file path or a directory path.
Cards will be combined into a single file or split apart into multiple
files as needed.  The behavior is as follows:

    input=file, output=file -> file to file
    input=dir,  output=file -> write all cards to one file
    input=file, output=dir  -> write to same-named file
    input=dir,  output=dir  -> write all files to same-named files

If input is a directory, vcf files in all sub-dirs will be found.
'''

GOOGLE_SCOPES = ['https://www.googleapis.com/auth/contacts']
GOOGLE_TOKEN_FILE_NAME = 'google_token.json'
GOOGLE_SECRET_FILE_NAME = 'google_client_secret.json'

GOOGLE_BATCH_CREATE_LIMIT = 200
GOOGLE_BATCH_DELETE_LIMIT = 500

VCard = vobject.base.Component
VCardProperty = vobject.base.ContentLine


debug = print
error = print
info = print
try:
    from rich import inspect
    from rich.pretty import pprint
except ImportError:
    inspect = debug
    from pprint import pprint


def fatal(*args, **kwargs) -> NoReturn:
    print(*args, **kwargs)
    sys.exit(1)


def chunks(seq: Sequence[Any], size: int) -> Iterator[list[Any]]:
    return [seq[pos : pos + size] for pos in range(0, len(seq), size)]


def read_vcards(file_path: str) -> list[VCard]:
    vcards = []
    with open(file_path, encoding='utf-8') as fp:
        components = vobject.readComponents(fp)
        for component in components:
            if component.name == 'VCARD':
                vcards.append(component)
    return vcards


def write_vcards(file_path: str, vcards: Sequence[VCard]) -> None:
    with open(file_path, 'w', newline='\n', encoding='utf-8') as fp:
        for vcard in vcards:
            vcard.serialize(fp)


def read_all_vcards(input_path: str) -> dict[str, list[VCard]]:
    '''Given a path, return a mapping from found vcf files to the list of VCard
    instances each vcf file contains.

    If input_path is a file, it will read just that file.
    If input_path is a directory, it will recursively read all found vcf files.
    '''

    def iter_input_dir(input_dir_path) -> Iterator[str]:
        assert os.path.isdir(input_dir_path)
        for dir_path, dir_names, file_names in os.walk(input_dir_path):
            for file_name in file_names:
                if os.path.splitext(file_name)[-1].lower() == '.vcf':
                    yield os.path.join(dir_path, file_name)

    def iter_input_file(input_file_path) -> Iterator[str]:
        yield input_file_path

    iter_func = iter_input_dir if os.path.isdir(input_path) else iter_input_file

    input_vcards = {}
    for input_file_path in iter_func(input_path):
        info(f'Reading "{input_file_path}"')
        input_vcards[input_file_path] = read_vcards(input_file_path)
    return input_vcards


def write_all_vcards(output_vcards: dict[str, list[VCard]], output_path: str) -> None:
    '''Given a mapping from input file paths to VCard (as returned by
    read_all_vcards), and an output path, write out the VCards.

    If output_path is a file, all VCards will be writtent o that file.
    If output_path is a directory, each set of VCards will be written to a
    file in the directory with a name that matches the filename it was
    originally read from.
    '''

    def iter_output_dir(
        output_vcards: dict[str, list[VCard]], output_dir_path: str
    ) -> Iterator[tuple(str, list[VCard])]:
        for input_file_path, vcards in output_vcards.items():
            input_file_name = os.path.basename(input_file_path)
            output_file_path = os.path.join(output_dir_path, input_file_name)
            yield (output_file_path, vcards)

    def iter_output_file(
        output_vcards: dict[str, list[VCard]], output_file_path: str
    ) -> Iterator[tuple(str, list[VCard])]:
        all_vcards = itertools.chain.from_iterable(output_vcards.values())
        yield (output_file_path, all_vcards)

    iter_func = iter_output_dir if os.path.isdir(output_path) else iter_output_file
    for output_file_path, vcards in iter_func(output_vcards, output_path):
        info(f'Writing "{output_file_path}"')
        write_vcards(output_file_path, vcards)


def get_single_param(params: dict[str, list[Any]], name: str) -> Any:
    if name in params:
        param_list = params.get(name)
        assert isinstance(param_list, list)
        assert len(param_list) == 1
        return param_list[0]
    return None


def get_multi_param(params: dict[str, list[Any, ...]], name: str) -> list[Any]:
    param = params.get(name, None)
    if param is None:
        return []
    else:
        return param if isinstance(param, list) else [param]


def get_vcard_photo_type(vcard: VCard) -> Optional[str]:
    '''Return one of:
    None
    'inline'
    'link'
    '''
    if hasattr(vcard, 'photo'):
        value = get_single_param(vcard.photo.params, 'VALUE')
        if value and 'uri' in value:
            return 'link'
        else:
            return 'inline'
    return None


def fix_card_photo(photo_prop: VCardProperty) -> bool:
    '''Constrain size/dimensions of a vcard photo according to the
    (undocumented?) rules that Apple applies to its contacts.
    '''
    MAX_IMAGE_DIM = 512
    MAX_BUFFER_SIZE = MAX_IMAGE_DIM * MAX_IMAGE_DIM // 4
    DEFAULT_QUALITY = 90
    QUALITY_BACKOFF = 5

    with io.BytesIO(photo_prop.value) as fp:
        original_image = Image.open(fp)

        max_dim = max(original_image.size)
        if max_dim <= MAX_IMAGE_DIM and len(photo_prop.value) <= MAX_BUFFER_SIZE:
            return False

        converted_image = original_image

        if converted_image.mode != 'RGB':
            converted_image = converted_image.convert('RGB')

        if max_dim > MAX_IMAGE_DIM:
            target_size = (
                int((converted_image.size[0] / max_dim) * MAX_IMAGE_DIM),
                int((converted_image.size[1] / max_dim) * MAX_IMAGE_DIM),
            )
            converted_image = converted_image.resize(target_size)

        quality = DEFAULT_QUALITY
        while True:
            compressed_bytes = io.BytesIO()
            converted_image.save(compressed_bytes, format='jpeg', optimize=True, quality=quality)
            photo_bytes = compressed_bytes.getvalue()
            if len(photo_bytes) > MAX_BUFFER_SIZE:
                if quality <= QUALITY_BACKOFF:
                    raise RuntimeError("Can't compress enough")
                else:
                    quality -= QUALITY_BACKOFF
            else:
                break

        photo_prop.value = photo_bytes
        while 'TYPE' in photo_prop.params:
            photo_prop.params.pop('TYPE')
        photo_prop.params['TYPE'] = ['jpeg']
        return True


class Person:
    '''Acts as a lowest-common-denominator bundle of data between the various
    contact stores (primarily the data I care about getting into Google contacts).
    '''

    def __init__(self, name: str, uid: Optional[str]):
        # This can be either the fn value from the vcard, or the org value if fn is blank
        assert isinstance(name, str)

        if uid is None:
            uid = str(uuid.uuid4())
        else:
            assert isinstance(uid, str)
            assert len(uid) > 0

        self.name = name
        self.uid = uid

        self.addresses: list[(str, Any), ...] = []
        self.email_addresses: list[(str, str), ...] = []
        self.nickname = ''
        self.note = ''
        self.organization = ''
        self.phone_numbers: list[(str, str), ...] = []
        self.photo_bytes = b''

    @staticmethod
    def from_vcard(vcard: VCard) -> Person:
        fullname = vcard.fn.value.strip()
        uid = vcard.uid.value.strip()
        person = Person(fullname, uid)

        for prop_name, prop_values in vcard.contents.items():
            assert isinstance(prop_values, list)
            assert len(prop_values) > 0
            match prop_name:
                case 'adr':
                    for address_prop in prop_values:
                        params = address_prop.params

                        address_types = get_multi_param(params, 'TYPE')
                        address_label = address_types[0].lower() if address_types else ''
                        assert isinstance(address_prop.value, vobject.vcard.Address)
                        address = str(address_prop.value)

                        person.addresses.append((address_label, address))
                case 'email':
                    for email_prop in prop_values:
                        params = email_prop.params

                        email_types = get_multi_param(params, 'TYPE')
                        email_label = email_types[0].lower() if email_types else ''
                        if email_label == 'pref':
                            email_label = ''
                        email = email_prop.value.strip()

                        person.email_addresses.append((email_label, email))
                case 'nickname':
                    assert len(prop_values) > 0
                    nickname_prop = prop_values[0]
                    assert isinstance(nickname_prop.value, str)
                    nickname = nickname_prop.value.strip()

                    person.nickname = nickname
                case 'note':
                    assert len(prop_values) == 1
                    note_prop = prop_values[0]
                    assert isinstance(note_prop.value, str)
                    note = note_prop.value.strip()

                    person.note = note
                case 'org':
                    assert len(prop_values) == 1
                    orgs_prop = prop_values[0]
                    assert isinstance(orgs_prop.value, list)
                    assert len(orgs_prop.value) > 0
                    org = orgs_prop.value[0].strip()

                    person.organization = org
                    if not person.name:
                        person.name = person.organization
                case 'photo':
                    assert len(prop_values) == 1
                    photo_prop = prop_values[0]

                    params = photo_prop.params
                    if 'VALUE' in params:
                        value = get_single_param(params, 'VALUE')
                        assert value == 'uri'
                        assert isinstance(photo_prop.value, str)
                        photo_uri = photo_prop.value.strip()
                    else:
                        encoding = get_single_param(params, 'ENCODING')
                        assert encoding is not None
                        assert encoding.lower() == 'b'

                        assert isinstance(photo_prop.value, bytes)
                        photo_bytes = photo_prop.value

                        person.photo_bytes = photo_bytes
                case 'tel':
                    for phone_prop in prop_values:
                        params = phone_prop.params

                        phone_types = get_multi_param(params, 'TYPE')
                        phone_label = phone_types[0].lower() if phone_types else ''
                        if phone_label == 'pref':
                            phone_label = ''
                        phone = phone_prop.value.strip()

                        person.phone_numbers.append((phone_label, phone))

        return person

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.name!r}, {self.uid!r})'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and hash(self) == hash(other)

    def __hash__(self) -> int:
        return hash(self.uid)


################################################################################


def get_google_credentials() -> Optional[Credentials]:
    creds = None
    token_file_path = os.path.join(THIS_DIR_PATH, GOOGLE_TOKEN_FILE_NAME)
    if os.path.exists(token_file_path):
        creds = Credentials.from_authorized_user_file(token_file_path, GOOGLE_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            secrets_file_path = os.path.join(THIS_DIR_PATH, GOOGLE_SECRET_FILE_NAME)
            flow = InstalledAppFlow.from_client_secrets_file(secrets_file_path, GOOGLE_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_file_path, 'w', encoding='utf-8') as fp:
            fp.write(creds.to_json())
    return creds


def hey_papi() -> googleapiclient.discovery.Resource:
    creds = get_google_credentials()
    if not creds:
        fatal('Unable to get Google credentials')
    service = googleapiclient.discovery.build('people', 'v1', credentials=creds)
    papi = service.people()
    return papi


def contact_info_from_person(person: Person) -> dict[str, Any]:
    '''Creates a dict that contains the info that Google People API can ingest.'''
    ci = {}

    ci['names'] = [{'unstructuredName': person.name}]
    ci['userDefined'] = [{'key': 'uid', 'value': person.uid}]

    ci['biographies'] = [{'value': person.note}]
    ci['nicknames'] = [{'value': person.nickname}]
    ci['organizations'] = [{'name': person.organization}]

    ci['addresses'] = []
    for label, address in person.addresses:
        ci['addresses'].append({'type': label, 'formattedValue': address})

    ci['emailAddresses'] = []
    for label, email_address in person.email_addresses:
        ci['emailAddresses'].append({'type': label, 'value': email_address})

    ci['phoneNumbers'] = []
    for label, phone_number in person.phone_numbers:
        ci['phoneNumbers'].append({'type': label, 'value': phone_number})

    return ci


def load_all_google_contacts(papi: googleapiclient.discovery.Resource) -> list[dict[str, Any]]:
    contacts = []
    page_token = ''
    connections = papi.connections()
    while page_token is not None:
        request = connections.list(
            resourceName='people/me',
            personFields='metadata,userDefined',
            pageToken=page_token,
        )
        response = request.execute()
        page_token = response.get('nextPageToken', None)
        contacts.extend(response.get('connections', []))
    return contacts


def delete_google_contacts(
    papi: googleapiclient.discovery.Resource, resource_names: Sequence[str]
) -> None:
    deleted_resource_names = []
    for resource_names_chunk in chunks(resource_names, GOOGLE_BATCH_DELETE_LIMIT):
        request = papi.batchDeleteContacts(body={'resourceNames': resource_names_chunk})
        response = request.execute()
        deleted_resource_names.extend(resource_names_chunk)
    return deleted_resource_names


def delete_all_google_contacts(papi: googleapiclient.discovery.Resource) -> None:
    contacts = load_all_google_contacts(papi)
    resource_names = [contact['resourceName'] for contact in contacts]
    return delete_google_contacts(papi, resource_names)


def create_google_contacts(
    papi: googleapiclient.discovery.Resource, persons: Sequence[Person]
) -> None:
    for ii, persons_chunk in enumerate(chunks(persons, GOOGLE_BATCH_CREATE_LIMIT)):
        start_index = ii * GOOGLE_BATCH_CREATE_LIMIT
        end_index = start_index + GOOGLE_BATCH_CREATE_LIMIT - 1
        info(f'Creating persons {start_index}-{end_index} of {len(persons)} total persons')
        contacts = [{'contactPerson': contact_info_from_person(person)} for person in persons_chunk]
        body = {'contacts': contacts, 'readMask': 'names'}
        request = papi.batchCreateContacts(body=body)
        response = request.execute()


################################################################################


def cmd_fullsync(args: argparse.Namespace) -> NoReturn:
    papi = hey_papi()
    info('Deleting contacts...', end='\n')
    deleted_resource_names = sorted(delete_all_google_contacts(papi))
    for name in deleted_resource_names:
        info(name)
    info(f'Deleted {len(deleted_resource_names)} contacts')
    info(f'Scraping contacts from "{args.contacts_root_dir}"')
    persons = [
        Person.from_vcard(vcard)
        for vcard in vcards
        for _, vcards in read_all_vcards(args.contacts_root_dir).items()
    ]
    info(f'Found {len(persons)} contacts')
    create_google_contacts(papi, persons)
    sys.exit(0)


def cmd_rewritecards(args: argparse.Namespace) -> NoReturn:
    input_vcards = read_all_vcards(args.input_path)
    write_all_vcards(input_vcards, args.output_path)
    sys.exit(0)


def cmd_fixphotos(args: argparse.Namespace) -> NoReturn:
    input_vcards = read_all_vcards(args.input_path)

    for input_file_path, vcards in input_vcards.items():
        for vcard in vcards:
            if get_vcard_photo_type(vcard) == 'inline':
                if fix_card_photo(vcard.photo):
                    info('optimized', vcard.fn.value)

    write_all_vcards(input_vcards, args.output_path)
    sys.exit(0)


def cmd_test(args: argparse.Namespace) -> NoReturn:
    for file_path, vcards in read_all_vcards(args.dir_path).items():
        for vcard in vcards:
            if get_vcard_photo_type(vcard) == 'link':
                uri = vcard.photo.value
                info(file_path, str(vcard.fn.value), vcard.uid.value, uri)
    sys.exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.set_defaults(func=lambda _: parser.print_usage(file=sys.stdout))
    subparsers = parser.add_subparsers()

    sp = subparsers.add_parser('fullsync', help='Force fresh upload of all contacts to Google')
    sp.add_argument('contacts_root_dir')
    sp.set_defaults(func=cmd_fullsync)

    sp = subparsers.add_parser(
        'rewritecards',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help='Read card(s) from input_path and write them out to output_path.',
        epilog=INPUT_OUTPUT_PATH_HELP,
    )
    sp.add_argument('input_path')
    sp.add_argument('output_path')
    sp.set_defaults(func=cmd_rewritecards)

    sp = subparsers.add_parser(
        'fixphotos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help='Fix photos in VCards to be compatible with iCloud',
        epilog=INPUT_OUTPUT_PATH_HELP,
    )
    sp.add_argument('input_path')
    sp.add_argument('output_path')
    sp.set_defaults(func=cmd_fixphotos)

    sp = subparsers.add_parser('test')
    sp.add_argument('dir_path')
    sp.set_defaults(func=cmd_test)

    args = parser.parse_args()
    return args


def main() -> NoReturn:
    ############################################################################
    # Map from a vcard UID to google resource name, if we ever want to mess
    # with gently updating existing contacts vs existing lazy scorched earth
    # delete-then-create method being used now.

    # contacts = load_all_google_contacts(papi)
    # uid_to_resource = {}
    # for contact in contacts:
    #     uid = None
    #     userDefined = contact.get('userDefined', [])
    #     for data in userDefined:
    #         if data.get('key') == 'uid':
    #             uid = data.get('value')
    #             break
    #     if uid:
    #         uid_to_resource[uid] = contact['resourceName']
    # all_uids = set([person.uid for person in persons])

    ############################################################################
    # Example of how to update a contact photo.  Non-batched, not worth sync'ing
    # to google for now.

    # for person in persons:
    #     ci = contact_info_from_person(person)
    #     request = papi.createContact(body=ci)
    #     response = request.execute()
    #     contact_resource = response.get('resourceName', None)
    #     if contact_resource and person.photo_bytes:
    #         encoded_photo = base64.b64encode(person.photo_bytes).decode('utf-8')
    #         request = papi.updateContactPhoto(
    #             resourceName=contact_resource, body={'photoBytes': encoded_photo}
    #         )
    #         response = request.execute()

    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

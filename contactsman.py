from __future__ import annotations
import argparse
import base64
from collections import defaultdict
from collections.abc import (
    Iterable,
    Sequence,
)
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

from rich import (
    inspect,
    traceback,
)
from rich.pretty import pprint

traceback.install(show_locals=True)

from PIL import Image
from google.auth.transport.requests import Request
import googleapiclient.discovery
import googleapiclient.errors
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
GOOGLE_UPDATE_PERSON_FIELDS = (
    'addresses',
    'biographies',
    'birthdays',
    'calendarUrls',
    'clientData',
    'emailAddresses',
    'events',
    'externalIds',
    'genders',
    'imClients',
    'interests',
    'locales',
    'locations',
    'memberships',
    'miscKeywords',
    'names',
    'nicknames',
    'occupations',
    'organizations',
    'phoneNumbers',
    'relations',
    'sipAddresses',
    'urls',
    'userDefined',
)
GOOGLE_LOAD_PERSON_FIELDS_ALL = (
    'addresses',
    'ageRanges',
    'biographies',
    'birthdays',
    'calendarUrls',
    'clientData',
    'coverPhotos',
    'emailAddresses',
    'events',
    'externalIds',
    'genders',
    'imClients',
    'interests',
    'locales',
    'locations',
    'memberships',
    'metadata',
    'miscKeywords',
    'names',
    'nicknames',
    'occupations',
    'organizations',
    'phoneNumbers',
    'photos',
    'relations',
    'sipAddresses',
    'skills',
    'urls',
    'userDefined',
)
GOOGLE_LOAD_PERSON_FIELDS_MINIMAL = (
    'metadata',
    'userDefined',
)

GOOGLE_BATCH_CREATE_LIMIT = 200
GOOGLE_BATCH_UPDATE_LIMIT = 200
GOOGLE_BATCH_DELETE_LIMIT = 500

VCard = vobject.base.Component
VCardProperty = vobject.base.ContentLine


def fatal(*args, **kwargs) -> NoReturn:
    print(*args, **kwargs)
    sys.exit(1)


def noop(*args, **kwargs) -> None:
    pass


debug = noop
error = print
info = print


def chunks(seq: Sequence[Any], size: int) -> Iterable[list[Any]]:
    return [seq[pos : pos + size] for pos in range(0, len(seq), size)]


def read_vcards(file_path: str) -> list[VCard]:
    vcards = []
    with open(file_path, encoding='utf-8') as fp:
        components = vobject.readComponents(fp)
        for component in components:
            if component.name == 'VCARD':
                vcards.append(component)
    return vcards


def write_vcards(file_path: str, vcards: Iterable[VCard]) -> None:
    with open(file_path, 'w', newline='\n', encoding='utf-8') as fp:
        for vcard in vcards:
            vcard.serialize(fp)


def read_all_vcards(input_path: str) -> dict[str, list[VCard]]:
    '''Given a path, return a mapping from found vcf files to the list of VCard
    instances each vcf file contains.

    If input_path is a file, it will read just that file.
    If input_path is a directory, it will recursively read all found vcf files.
    '''
    info(f'Reading vCards from "{input_path}"...')

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
        debug(f'Reading "{input_file_path}"')
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
    info(f'Writing vCards to "{output_path}"...')

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
        debug(f'Writing "{output_file_path}"')
        write_vcards(output_file_path, vcards)


def flatten_vcards_list(all_vcards):
    '''Return a simple list of VCards given a dict returned by read_all_vcards.'''
    return [
        vcard for vcard in itertools.chain.from_iterable([vcards for vcards in all_vcards.values()])
    ]


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
        self.rev = ''

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
                case 'rev':
                    assert len(prop_values) == 1
                    rev_prop = prop_values[0]
                    assert isinstance(rev_prop.value, str)
                    rev = rev_prop.value.strip()

                    person.rev = rev
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
    ci['userDefined'] = [
        {'key': 'uid', 'value': person.uid},
        {'key': 'rev', 'value': person.rev},
    ]

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


def execute_request(request: googleapiclient.http.HttpRequest) -> Any:
    return request.execute(num_retries=3)
    # arbitrary, and number of tries is (len(backoff_times) + 1)
    # backoff_times = (
    #     0.1,
    #     0.5,
    #     2.0,
    # )
    # for ii, backoff_time in enumerate(backoff_times):
    #     try:
    #         result = request.execute()
    #         return result
    #     except googleapiclient.errors.HttpError as ex:
    #         if ex.resp.status == 503:
    #             if ii == len(backoff_times):
    #                 raise
    #             time.sleep(backoff_time)


def load_all_google_contacts(
    papi: googleapiclient.discovery.Resource,
    person_fields: Iterable[str] = GOOGLE_LOAD_PERSON_FIELDS_MINIMAL,
) -> list[dict[str, Any]]:
    info(f'Loading all Google contacts...')

    contacts = []
    page_token = ''
    connections = papi.connections()
    while page_token is not None:
        request = connections.list(
            resourceName='people/me',
            personFields=','.join(person_fields),
            pageToken=page_token,
        )
        response = execute_request(request)
        page_token = response.get('nextPageToken', None)
        contacts.extend(response.get('connections', []))
    return contacts


def delete_google_contacts(
    papi: googleapiclient.discovery.Resource, resource_names: Iterable[str]
) -> None:
    deleted_resource_names = []
    for resource_names_chunk in chunks(resource_names, GOOGLE_BATCH_DELETE_LIMIT):
        request = papi.batchDeleteContacts(body={'resourceNames': resource_names_chunk})
        response = execute_request(request)
        deleted_resource_names.extend(resource_names_chunk)
    return deleted_resource_names


def delete_all_google_contacts(papi: googleapiclient.discovery.Resource) -> None:
    info('Deleting all Google contacts...')
    contacts = load_all_google_contacts(papi)
    resource_names = [contact['resourceName'] for contact in contacts]
    return delete_google_contacts(papi, resource_names)


def create_google_contacts(
    papi: googleapiclient.discovery.Resource, persons: Iterable[Person]
) -> None:
    for ii, persons_chunk in enumerate(chunks(persons, GOOGLE_BATCH_CREATE_LIMIT)):
        contacts = [{'contactPerson': contact_info_from_person(person)} for person in persons_chunk]
        start_index = ii * GOOGLE_BATCH_CREATE_LIMIT
        end_index = start_index + len(contacts) - 1
        debug(f'Creating persons {start_index}-{end_index} of {len(persons)} total persons')
        body = {'contacts': contacts, 'readMask': 'names'}
        request = papi.batchCreateContacts(body=body)
        response = execute_request(request)


def update_google_contacts(
    papi: googleapiclient.discovery.Resource, contacts: dict[str, dict[Any, Any]]
) -> None:
    for ii, resource_contacts_chunk in enumerate(
        chunks(tuple(contacts.items()), GOOGLE_BATCH_UPDATE_LIMIT)
    ):
        contacts_chunk = dict(resource_contacts_chunk)
        start_index = ii * GOOGLE_BATCH_UPDATE_LIMIT
        end_index = start_index + len(contacts_chunk) - 1
        debug(f'Updating persons {start_index}-{end_index} of {len(contacts_chunk)} total persons')
        body = {
            'contacts': contacts_chunk,
            'updateMask': ','.join(GOOGLE_UPDATE_PERSON_FIELDS),
        }
        request = papi.batchUpdateContacts(body=body)
        response = execute_request(request)


################################################################################


def cmd_updategoogle(args: argparse.Namespace) -> NoReturn:
    papi = hey_papi()

    all_resources = []
    if args.force:
        deleted_resource_names = sorted(delete_all_google_contacts(papi))
        for name in deleted_resource_names:
            debug(name)
        info(f'Deleted {len(deleted_resource_names)} contacts')
    else:
        all_resources = load_all_google_contacts(papi)
        info(f'Loaded {len(all_resources)} Google contacts')

    all_persons = [
        Person.from_vcard(vcard)
        for vcard in flatten_vcards_list(read_all_vcards(args.vcards_root_dir))
    ]
    info(f'Read {len(all_persons)} vCards')

    uid_to_person = {person.uid: person for person in all_persons}
    uid_to_resource = {}
    orphaned_resources = []
    stale_person_uids = []
    for contact in all_resources:
        resource_name = contact['resourceName']
        uid = None
        rev = None
        userDefined = contact.get('userDefined', [])
        for data in userDefined:
            match data.get('key'):
                case 'uid':
                    uid = data.get('value')
                case 'rev':
                    rev = data.get('value')
        if uid:
            person = uid_to_person.get(uid, None)
            if person is None:
                orphaned_resources.append(resource_name)
            else:
                if rev is None or rev != person.rev:
                    stale_person_uids.append(uid)
            uid_to_resource[uid] = contact
        else:
            orphaned_resources.append(resource_name)

    new_persons = [
        uid_to_person[person.uid] for person in all_persons if person.uid not in uid_to_resource
    ]
    info(f'Creating {len(new_persons)} new Google contacts')
    create_google_contacts(papi, new_persons)

    update_resources = {}
    for uid in stale_person_uids:
        person = uid_to_person[uid]
        resource = uid_to_resource[uid]
        contact_info = contact_info_from_person(person)
        contact_info['etag'] = resource['etag']
        contact_info['memberships'] = [
            {'contactGroupMembership': {'contactGroupResourceName': 'contactGroups/myContacts'}}
        ]
        update_resources[resource['resourceName']] = contact_info
    info(f'Updating {len(update_resources)} Google contacts')
    update_google_contacts(papi, update_resources)

    info(f'Deleting {len(orphaned_resources)} orphaned Google contacts')
    delete_google_contacts(papi, orphaned_resources)

    sys.exit(0)


def cmd_rewritecards(args: argparse.Namespace) -> NoReturn:
    input_vcards = read_all_vcards(args.input_path)
    write_all_vcards(input_vcards, args.output_path)
    sys.exit(0)


def cmd_fixphotos(args: argparse.Namespace) -> NoReturn:
    input_vcards = read_all_vcards(args.input_path)

    numchanged = 0
    for input_file_path, vcards in input_vcards.items():
        for vcard in vcards:
            if get_vcard_photo_type(vcard) == 'inline':
                if fix_card_photo(vcard.photo):
                    info('optimized', vcard.fn.value)
                    numchanged += 1
    info(f'Updated {numchanged} vCards')

    if numchanged > 0:
        write_all_vcards(input_vcards, args.output_path)
    sys.exit(0)


def cmd_newuids(args: argparse.Namespace) -> NoReturn:
    input_vcards = read_all_vcards(args.input_path)

    numchanged = 1
    for input_file_path, vcards in input_vcards.items():
        for vcard in vcards:
            olduid = vcard.uid.value.strip()
            newuid = str(uuid.uuid4())
            vcard.uid.value = newuid
            debug(f'{olduid} -> {newuid}')
            numchanged += 1
    info(f'Updated {numchanged} vCards')

    write_all_vcards(input_vcards, args.output_path)
    sys.exit(0)


def cmd_test(args: argparse.Namespace) -> NoReturn:
    papi = hey_papi()
    connections = papi.connections()
    request = connections.list(
        resourceName='people/me',
        personFields=','.join(GOOGLE_LOAD_PERSON_FIELDS_MINIMAL),
    )
    response = execute_request(request)
    pprint(response)
    sys.exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.set_defaults(func=lambda _: parser.print_usage(file=sys.stdout))
    subparsers = parser.add_subparsers()

    sp = subparsers.add_parser('updategoogle', help='Update google contacts from vCards')
    sp.add_argument('vcards_root_dir')
    sp.add_argument(
        '-f',
        '--force',
        action='store_true',
        help='Delete all Google contacts first and force a full re-upload',
    )
    sp.set_defaults(func=cmd_updategoogle)

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
        help='Fix photos in vCards to be compatible with iCloud',
        epilog=INPUT_OUTPUT_PATH_HELP,
    )
    sp.add_argument('input_path')
    sp.add_argument('output_path')
    sp.set_defaults(func=cmd_fixphotos)

    sp = subparsers.add_parser(
        'newuids',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help='Give all vCards a new UID',
        epilog=INPUT_OUTPUT_PATH_HELP,
    )
    sp.add_argument('input_path')
    sp.add_argument('output_path')
    sp.set_defaults(func=cmd_newuids)

    sp = subparsers.add_parser('test')
    sp.set_defaults(func=cmd_test)

    args = parser.parse_args()
    return args


def main() -> NoReturn:
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
    if args.verbose:
        global debug
        debug = info
    args.func(args)


if __name__ == '__main__':
    main()

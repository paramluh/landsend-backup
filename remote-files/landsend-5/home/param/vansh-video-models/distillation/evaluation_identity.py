"""Distinguish saved checkpoints from evaluations of live model state."""
import uuid


def evaluation_identity_fields(value):
    if value.get('evaluation_kind') == 'in_memory':
        identity = value.get('evaluation_id')
        try:
            valid_uuid = isinstance(identity, str) and uuid.UUID(identity).hex == identity.replace('-', '').lower()
        except (ValueError, AttributeError):
            valid_uuid = False
        if not valid_uuid or 'checkpoint_id' not in value or value['checkpoint_id'] is not None:
            raise ValueError('In-memory evaluation requires a UUID and checkpoint_id=null')
        for key in ('student_updates', 'rounds'):
            if type(value.get(key)) is not int or value[key] < 0:
                raise ValueError('In-memory evaluation requires completed training counters')
        if value['rounds'] < value['student_updates']:
            raise ValueError('In-memory evaluation counters are inconsistent')
        return {key: value[key] for key in
                ('checkpoint_id', 'evaluation_id', 'evaluation_kind', 'student_updates', 'rounds')}
    if value.get('evaluation_kind') not in (None, 'checkpoint') or not value.get('checkpoint_id'):
        raise ValueError('Evaluation requires an explicit checkpoint or in-memory identity')
    return {'checkpoint_id': str(value['checkpoint_id'])}

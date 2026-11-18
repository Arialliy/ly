"""Bind the eligible paired manifest before any new endpoint readouts."""
import time
import runtime as rt


def main():
    destination = rt.E / 'PROTOCOL_FREEZE.json'
    if destination.exists():
        raise FileExistsError(destination)
    previous = rt.ROOT / 'experiments/c_effective_update_v1'
    old = rt.read_json(previous / 'PROTOCOL_FREEZE.json')
    sources = dict(old['reused_sources'])
    sources.update({str(previous / name): value for name, value in old['files'].items()})
    sources[str(previous / 'PROTOCOL_FREEZE.json')] = rt.sha(previous / 'PROTOCOL_FREEZE.json')
    for name in ['OPERATOR_CHECK.json', 'COMPLETE.json', 'VALIDATION.json']:
        sources[str(previous / name)] = rt.sha(previous / name)
    plan = rt.read_json(rt.E / 'PAIR_PLAN.json')
    sources.update(plan['source_sha256'])
    for name, expected in sources.items():
        assert rt.sha(name) == expected, name
    required = ['PROTOCOL.md', 'PAIR_PLAN.json', 'PAIR_MANIFEST.csv',
                'ELIGIBILITY_AUDIT.json', 'GATE_SPEC.json', 'evaluation/MANIFEST_CHECK.json']
    files = {name: rt.sha(rt.E / name) for name in required}
    for folder in ('code', 'replay'):
        for path in sorted((rt.E / folder).iterdir()):
            if path.is_file() and path.suffix in ('.py', '.json', '.npz'):
                files[str(path.relative_to(rt.E))] = rt.sha(path)
    for path in sorted((rt.E / 'evaluation').glob('*REVIEW.json')):
        files[str(path.relative_to(rt.E))] = rt.sha(path)
    rt.save_json(destination, {'status': 'FROZEN_BEFORE_NEW_OBSERVER_READOUTS',
        'created_unix': time.time(), 'files': files, 'reused_sources': sources,
        'initial_checkpoint_sha256': rt.sha(rt.INIT_CP), 'initial_step': 2304,
        'common_old_pairs': plan['common_old_pair_ids'], 'added_pairs': plan['added_pair_ids'],
        'pairs': len(plan['pairs']), 'new_optimizer_steps': 0,
        'max_user_authorized_new_optimizer_steps': 27, 'uses_development': False,
        'copies_accumulated': False, 'conditional_config_only': True,
        'no_third_supplement': True})
    print({'status': 'FROZEN', 'sha256': rt.sha(destination), 'files': len(files),
           'reused_sources': len(sources), 'pairs': len(plan['pairs'])}, flush=True)


if __name__ == '__main__':
    main()

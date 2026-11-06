"""Read-only TurtleBot diagnostics and Zenoh bridge routes."""
import json
from pathlib import Path
import time
from transport import ZenohLink, TYPES

received, admin = {}, []
with ZenohLink('127.0.0.1:7447') as link:
    for key, kind in [('tb3/diagnostics', 'diagnostic_msgs/msg/DiagnosticArray'),
                      ('tb3/joint_states', 'sensor_msgs/msg/JointState')]:
        def decode(raw, topic=key, msgtype=kind):
            message = TYPES.deserialize_cdr(raw, msgtype)
            if topic.endswith('diagnostics'):
                received[topic] = [{'name': item.name, 'level': int(item.level), 'message': item.message,
                                    'values': {v.key: v.value for v in item.values}} for item in message.status]
            else:
                received[topic] = {'name': message.name, 'position': message.position.tolist(),
                                   'velocity': message.velocity.tolist()}
            return None
        link.subscribe(key, decode)
    for reply in link.session.get('@/**', timeout=2):
        try:
            sample = reply.ok
            payload = sample.payload.to_bytes().decode(errors='replace')
            key = str(sample.key_expr)
            if any(word in key + payload for word in ('tb3', 'cmd_vel', 'motor_power')):
                admin.append({'key': key, 'payload': payload})
        except Exception:
            pass
    time.sleep(3)
result = {'topics': received, 'routes': admin}
(Path(__file__).resolve().parent/'logs/waffle_manual_diagnostics.json').write_text(json.dumps(result, indent=2))
print(json.dumps(received, indent=2))
for entry in admin[:12]:
    print(entry['key'], entry['payload'][:1600])
print(f'Matching admin responses: {len(admin)}')

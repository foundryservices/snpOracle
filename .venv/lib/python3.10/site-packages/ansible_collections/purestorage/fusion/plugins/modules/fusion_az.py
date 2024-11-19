#!/usr/bin/python
# -*- coding: utf-8 -*-

# (c) 2022, Simon Dodsley (simon@purestorage.com)
# GNU General Public License v3.0+ (see COPYING.GPLv3 or https://www.gnu.org/licenses/gpl-3.0.txt)

from __future__ import absolute_import, division, print_function

__metaclass__ = type

DOCUMENTATION = r"""
---
module: fusion_az
version_added: '1.0.0'
short_description:  Create Availability Zones in Pure Storage Fusion
description:
- Manage an Availability Zone in Pure Storage Fusion.
author:
- Pure Storage Ansible Team (@sdodsley) <pure-ansible-team@purestorage.com>
notes:
- Supports C(check mode).
options:
  name:
    description:
    - The name of the Availability Zone.
    type: str
    required: true
  state:
    description:
    - Define whether the Availability Zone should exist or not.
    default: present
    choices: [ present, absent ]
    type: str
  display_name:
    description:
    - The human name of the Availability Zone.
    - If not provided, defaults to I(name).
    type: str
  region:
    description:
    - Region within which the AZ is created.
    type: str
extends_documentation_fragment:
- purestorage.fusion.purestorage.fusion
"""

EXAMPLES = r"""
- name: Create new AZ foo
  purestorage.fusion.fusion_az:
    name: foo
    display_name: "foo AZ"
    app_id: key_name
    key_file: "az-admin-private-key.pem"

- name: Delete AZ foo
  purestorage.fusion.fusion_az:
    name: foo
    state: absent
    app_id: key_name
    key_file: "az-admin-private-key.pem"
"""

RETURN = r"""
"""

HAS_FUSION = True
try:
    import fusion as purefusion
except ImportError:
    HAS_FUSION = False

from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
    get_fusion,
    fusion_argument_spec,
)


def get_az(module, fusion):
    """Get Availability Zone or None"""
    az_api_instance = purefusion.AvailabilityZonesApi(fusion)
    try:
        return az_api_instance.get_availability_zone(
            availability_zone_name=module.params["name"],
            region_name=module.params["region"],
        )
    except purefusion.rest.ApiException:
        return None


def get_region(module, fusion):
    """Get Region or None"""
    region_api_instance = purefusion.RegionsApi(fusion)
    try:
        return region_api_instance.get_region(
            region_name=module.params["region"],
        )
    except purefusion.rest.ApiException:
        return None


def delete_az(module, fusion):
    """Delete Availability Zone"""

    az_api_instance = purefusion.AvailabilityZonesApi(fusion)

    changed = True
    if not module.check_mode:
        try:
            az_api_instance.delete_availability_zone(
                region_name=module.params["region"],
                availability_zone_name=module.params["name"],
            )
        except purefusion.rest.ApiException as err:
            module.fail_json(
                msg="Availability Zone {0} deletion failed.: {1}".format(
                    module.params["name"], err
                )
            )

    module.exit_json(changed=changed)


def create_az(module, fusion):
    """Create Availability Zone"""

    az_api_instance = purefusion.AvailabilityZonesApi(fusion)

    changed = True
    if not module.check_mode:
        if not module.params["display_name"]:
            display_name = module.params["name"]
        else:
            display_name = module.params["display_name"]
        try:
            azone = purefusion.AvailabilityZonePost(
                name=module.params["name"],
                display_name=display_name,
            )
            az_api_instance.create_availability_zone(
                azone, region_name=module.params["region"]
            )
        except purefusion.rest.ApiException as err:
            module.fail_json(
                msg="Availability Zone {0} creation failed.: {1}".format(
                    module.params["name"], err
                )
            )

    module.exit_json(changed=changed)


def main():
    """Main code"""
    argument_spec = fusion_argument_spec()
    argument_spec.update(
        dict(
            name=dict(type="str", required=True),
            display_name=dict(type="str"),
            region=dict(type="str"),
            state=dict(type="str", default="present", choices=["present", "absent"]),
        )
    )

    module = AnsibleModule(argument_spec, supports_check_mode=True)

    fusion = get_fusion(module)
    state = module.params["state"]
    azone = get_az(module, fusion)
    if not get_region(module, fusion):
        module.fail_json(
            msg="Region {0} does not exist.".format(module.params["region"])
        )

    if not azone and state == "present":
        create_az(module, fusion)
    elif azone and state == "absent":
        delete_az(module, fusion)
    else:
        module.exit_json(changed=False)

    module.exit_json(changed=False)


if __name__ == "__main__":
    main()

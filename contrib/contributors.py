#!/usr/bin/env python3
"""Update the contributor list

Alias handling is done by git with .mailmap
New contributors are merged in the short-term list.
Moving them to a "higher" list should be a manual process.
"""

import re
from pathlib import Path
import subprocess

CONTRIB_FILE_CONTRIBUTOR_RE = re.compile(r'\* (.+) (<.+>)')


def format_contributor(contributor):
    return " * {0} {1}".format(contributor["name"], contributor["email"])


def get_contributors():
    """ Parse all contributors from output of git shortlog -se
    """
    contributors = []
    p_git = subprocess.run(["git", "shortlog", "-se"], stdout=subprocess.PIPE)
    for line in p_git.stdout.decode('utf-8').split('\n'):
        m = re.search(r"(\d+)\t(.+) (<.+>)", line)
        if m:
            contributors.append({
                "count": int(m.group(1)),
                "name": m.group(2),
                "email": m.group(3)
            })
    return contributors


def get_old_contributors(contrib_file_lines):
    """ Parse existing contributors from CONTRIBUTORS.rst

    Returns:
        (list) Contributors as {"name", "email"} dicts
    """
    old_contributors = []
    for line in contrib_file_lines:
        m = CONTRIB_FILE_CONTRIBUTOR_RE.search(line)
        if m:
            old_contributors.append({"name": m.group(1), "email": m.group(2)})
    return old_contributors


def get_new_contributors(contributors, old_contributors):
    """ Find new contributors and any possible alias or email changes

    Returns:
        (tuple) list of new contributors,
                list of new aliases as (contributor, existing_name),
                list of new emails as (contributor, existing_email)
    """
    old_email_names = {c['email']: c['name'] for c in old_contributors}
    old_name_emails = {c['name']: c['email'] for c in old_contributors}
    new_contributors = []
    new_alias = []
    new_email = []
    for contributor in contributors:
        name, email = contributor['name'], contributor['email']
        existing_name, existing_email = old_email_names.get(email), old_name_emails.get(name)

        if existing_name == name and existing_email == email:
            # exact combination already in list
            pass
        elif existing_name is None and existing_email is None:
            new_contributors.append(contributor)
        elif existing_name is not None:
            new_alias.append((contributor, existing_name))
        elif existing_email is not None:
            new_email.append((contributor, existing_email))
    return (
        sorted(new_contributors, key=lambda x: x['name'].split()[-1].lower()),
        new_alias,
        new_email
    )


def merge_short_term_contributors(contrib_file_lines, new_contributors):
    """ Merge new contributors into Short-term Contributions section in
    alphabetical order.

    Returns:
        (list) Lines including new contributors for writing to CONTRIBUTORS.rst
    """
    short_term_found = False
    for (i, line) in enumerate(contrib_file_lines):
        if not short_term_found:
            if "Short-term" in line:
                short_term_found = True
        else:
            if CONTRIB_FILE_CONTRIBUTOR_RE.search(line):
                break

    short_term_contributor_lines = [l for l in contrib_file_lines[i:] if l] + \
        [format_contributor(c) + "\n" for c in new_contributors]

    def last_name_sort(contrib_line):
        m = CONTRIB_FILE_CONTRIBUTOR_RE.search(contrib_line)
        return m.group(1).split()[-1].lower()

    return contrib_file_lines[:i] + sorted(short_term_contributor_lines, key=last_name_sort)


def main():
    contrib_file = Path("CONTRIBUTORS.rst")
    with contrib_file.open() as f:
        contrib_file_lines = f.readlines()

    old_contributors = get_old_contributors(contrib_file_lines)

    contributors = get_contributors()
    new_contributors, new_alias, new_email = get_new_contributors(contributors, old_contributors)

    for contributor, old_name in new_alias:
        print("new alias {0} for {1} {2} ?".format(
            contributor['name'], old_name, contributor['email']))

    for contributor, old_email in new_email:
        print("new email {0} for {1} {2} ?".format(
            contributor['email'], contributor['name'], old_email))

    if new_alias or new_email:
        print("Please update .mailmap")

    if new_contributors:
        print("inserting:")
        print("\n".join([format_contributor(c) for c in new_contributors]))

    with contrib_file.open("w") as f:
        f.writelines(merge_short_term_contributors(contrib_file_lines, new_contributors))


if __name__ == "__main__":
    main()

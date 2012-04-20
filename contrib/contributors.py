#!/usr/bin/python2
"""Update the contributor list

Alias handling is done by git with .mailmap
New contributors are merged in the short-term list.
Moving them to a "higher" list should be a manual process.
"""

import fileinput
from subprocess import Popen, PIPE

def format_contributor(contributor):
    return " * {0} {1}".format(
            " ".join(contributor["name"]),
            contributor["email"])


def main():
    # generate list of contributors
    contributors = []
    p_git = Popen(["git", "shortlog", "-se"], stdout=PIPE)
    for line in p_git.stdout:
        contributors.append({
            'count': int(line.split("\t")[0].strip()),
            'name': line.split("\t")[1].split()[0:-1],
            'email': line.split("\t")[1].split()[-1]
            })

    # cache listed contributors
    old_contributors = []
    with open("CONTRIBUTORS.rst", "r") as contrib_file:
        for line in contrib_file:
            if "@" in line:
                old_contributors.append({
                    'name': line.split()[1:-1],
                    'email': line.split()[-1]
                    })

    old = map(lambda x: (x['name'], x['email']), old_contributors)
    old_emails = map(lambda x: x['email'], old_contributors)
    old_names = map(lambda x: x['name'], old_contributors)

    # check which contributors are new
    new_contributors = []
    update_mailmap = False
    for contributor in contributors:
        if (contributor['name'], contributor['email']) in old:
            # this exact combination already in the list
            pass
        elif (contributor['email'] not in old_emails
                and contributor['name'] not in old_names):
            # name AND email are not in the list
            new_contributors.append(contributor)
        elif contributor['email'] in old_emails:
            # email is listed, but with another name
            old_name = filter(lambda x: x['email'] == contributor['email'],
                    old_contributors)[0]['name']
            print "new alias %s for %s %s ?" % (
                    " ".join(contributor['name']),
                    " ".join(old_name),
                    contributor['email'])
            update_mailmap = True
        elif contributor['name'] in old_names:
            # probably a new email for a previous contributor
            other_mail = filter(lambda x: x['name'] == contributor['name'],
                old_contributors)[0]['email']
            print "new email %s for %s %s ?" % (
                contributor['email'],
                " ".join(contributor['name']),
                other_mail)
            update_mailmap = True
    if update_mailmap:
        print "Please update .mailmap"

    # sort on the last word of the name
    new_contributors = sorted(new_contributors,
            key=lambda x: x['name'][-1].lower())

    # show new contributors to be merged to the list
    if new_contributors:
        print "inserting:"
        for contributor in new_contributors:
            print format_contributor(contributor)

    # merge with alphabetical (by last part of name) contributor list
    i = 0
    short_term_found = False
    for line in fileinput.input("CONTRIBUTORS.rst", inplace=1):
        if not short_term_found:
            print line,
            if "Short-term" in line:
                short_term_found = True
        else:
            if i >= len(new_contributors) or "@" not in line:
                print line,
            else:
                listed_name = line.split()[-2].lower()
                contributor = new_contributors[i]
                # insert all new contributors that fit here
                while listed_name > contributor["name"][-1].lower():
                    print format_contributor(contributor)
                    i += 1
                    if i < len(new_contributors):
                        contributor = new_contributors[i]
                    else:
                        break
                print line,
    # append remaining contributors
    with open("CONTRIBUTORS.rst", "a") as contrib_file:
        while i < len(new_contributors):
            contrib_file.write(format_contributor(new_contributors[i]) + "\n")
            i += 1


if __name__ == "__main__":
    main()

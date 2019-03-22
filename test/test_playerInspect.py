import unittest
from io import StringIO
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch, MagicMock

import contrib.playerInspect as player_inspect


class TestPlayerInspect(unittest.TestCase):
    def setUp(self):
        self.player_data = {
            'AbsorptionAmount': 0.0,
            'Air': 300,
            'Attributes': [
                {'Base': 20.0, 'Name': 'generic.maxHealth'},
                {'Base': 0.0, 'Name': 'generic.knockbackResistance'},
                {'Base': 0.10000000149011612, 'Name': 'generic.movementSpeed'},
                {'Base': 0.0, 'Name': 'generic.armor'},
                {'Base': 0.0, 'Name': 'generic.armorToughness'},
                {'Base': 1.0, 'Name': 'generic.attackDamage'},
                {'Base': 4.0, 'Name': 'generic.attackSpeed'},
                {'Base': 0.0, 'Name': 'generic.luck'}
            ],
            'DataVersion': 1631,
            'DeathTime': 0,
            'Dimension': 0,
            'EnderItems': [],
            'FallDistance': 0.0,
            'FallFlying': 0,
            'Fire': -20,
            'Health': 20.0,
            'HurtByTimestamp': 0,
            'HurtTime': 0,
            'Inventory': [{'Count': 1, 'Slot': -106, 'id': 'minecraft:sign'}],
            'Invulnerable': 0,
            'Motion': [0.0, -0.0784000015258789, 0.0],
            'OnGround': 1,
            'PortalCooldown': 0,
            'Pos': [-96.11859857363737, 70.0, -44.17768261916891],
            'Rotation': [-72.00011444091797, 38.250030517578125],
            'Score': 0,
            'SelectedItemSlot': 0,
            'SleepTimer': 0,
            'Sleeping': 0,
            "SpawnX": 10,
            "SpawnY": 52,
            "SpawnZ": 10,
            'UUIDLeast': -7312926203658200544,
            'UUIDMost': 6651100054519957107,
            'XpLevel': 0,
            'XpP': 0.0,
            'XpSeed': 0,
            'XpTotal': 0,
            'abilities': {
                'flySpeed': 0.05000000074505806,
                'flying': 0,
                'instabuild': 1,
                'invulnerable': 1,
                'mayBuild': 1,
                'mayfly': 1,
                'walkSpeed': 0.10000000149011612
            },
            'foodExhaustionLevel': 0.0,
            'foodLevel': 20,
            'foodSaturationLevel': 5.0,
            'foodTickTimer': 0,
            'playerGameType': 1,
            'recipeBook': {
                'isFilteringCraftable': 0,
                'isFurnaceFilteringCraftable': 0,
                'isFurnaceGuiOpen': 0,
                'isGuiOpen': 0,
                'recipes': [],
                'toBeDisplayed': []
            },
            'seenCredits': 0
        }

    @patch('sys.stdout', new_callable=StringIO)
    def test_print_player(self, mock_stdout):
        expected = "\n".join([
            "Position:\t-96, 70, -44\t(dim: 0)",
            "Spawn:\t\t10, 52, 10",
            "Health:\t20\tLevel:\t\t0\t\tGameType:\t1",
            "Food:\t20\tTotal XP:\t0",
            "Inventory: 1 items",
            "  1   minecraft:sign\n"])

        player_inspect.print_player(self.player_data)

        self.assertEqual(mock_stdout.getvalue(), expected)

    @patch('sys.stdout', new_callable=StringIO)
    def test_print_player_no_spawn(self, mock_stdout):
        expected = "\n".join([
            "Position:\t-96, 70, -44\t(dim: 0)",
            "Health:\t20\tLevel:\t\t0\t\tGameType:\t1",
            "Food:\t20\tTotal XP:\t0",
            "Inventory: 1 items",
            "  1   minecraft:sign\n"])

        player_data = {
            k: v for k, v in self.player_data.items()
            if k not in("SpawnX", "SpawnY", "SpawnZ")
        }
        player_inspect.print_player(player_data)

        self.assertEqual(mock_stdout.getvalue(), expected)

    @patch('sys.stdout', new_callable=StringIO)
    def test_print_player_sub_entry(self, mock_stdout):
        expected = "\n".join([
            "\tPosition:\t-96, 70, -44\t(dim: 0)",
            "\tSpawn:\t\t10, 52, 10",
            "\tHealth:\t20\tLevel:\t\t0\t\tGameType:\t1",
            "\tFood:\t20\tTotal XP:\t0",
            "\tInventory: 1 items\n"])

        player_inspect.print_player(self.player_data, sub_entry=True)

        self.assertEqual(mock_stdout.getvalue(), expected)

    @patch('sys.stdout', new_callable=StringIO)
    def test_print_player_sub_entry_no_spawn(self, mock_stdout):
        expected = "\n".join([
            "\tPosition:\t-96, 70, -44\t(dim: 0)",
            "\tHealth:\t20\tLevel:\t\t0\t\tGameType:\t1",
            "\tFood:\t20\tTotal XP:\t0",
            "\tInventory: 1 items\n"])

        player_data = {
            k: v for k, v in self.player_data.items()
            if k not in("SpawnX", "SpawnY", "SpawnZ")
        }
        player_inspect.print_player(player_data, sub_entry=True)

        self.assertEqual(mock_stdout.getvalue(), expected)

    def test_find_all_player_files(self):
        dir_path = MagicMock(Path)
        files = [Path('def0492d-0fe9-43ff-a3d5-8c3fc9160c94.dat'),
                 Path('074c808a-1f04-4bdd-8385-bd74601210a1.dat'),
                 Path('104e149d-a802-4a27-ac8f-ceab5279087c.dat')]
        dir_path.iterdir.return_value = (f for f in files)

        expected = [(Path('def0492d-0fe9-43ff-a3d5-8c3fc9160c94.dat'),
                     'def0492d-0fe9-43ff-a3d5-8c3fc9160c94'),
                    (Path('074c808a-1f04-4bdd-8385-bd74601210a1.dat'),
                     '074c808a-1f04-4bdd-8385-bd74601210a1'),
                    (Path('104e149d-a802-4a27-ac8f-ceab5279087c.dat'),
                     '104e149d-a802-4a27-ac8f-ceab5279087c')]
        result = player_inspect.find_all_player_files(dir_path)
        self.assertListEqual(list(result), expected)

    def test_find_player_file(self):
        dir_path = MagicMock(Path)
        files = [Path('def0492d-0fe9-43ff-a3d5-8c3fc9160c94.dat'),
                 Path('074c808a-1f04-4bdd-8385-bd74601210a1.dat'),
                 Path('104e149d-a802-4a27-ac8f-ceab5279087c.dat')]
        dir_path.iterdir.return_value = (f for f in files)

        expected = (Path('104e149d-a802-4a27-ac8f-ceab5279087c.dat'),
                    '104e149d-a802-4a27-ac8f-ceab5279087c')
        result = player_inspect.find_player_file(
            dir_path, selected_player='104e149d-a802-4a27-ac8f-ceab5279087c')
        self.assertEqual(result, expected)

    def test_find_player_file_raises_when_selected_player_not_found(self):
        dir_path = MagicMock(Path)
        files = [Path('def0492d-0fe9-43ff-a3d5-8c3fc9160c94.dat'),
                 Path('104e149d-a802-4a27-ac8f-ceab5279087c.dat')]
        dir_path.iterdir.return_value = (f for f in files)

        with self.assertRaises(FileNotFoundError):
            player_inspect.find_player_file(dir_path, selected_player='NON_EXISTENT_UUID')

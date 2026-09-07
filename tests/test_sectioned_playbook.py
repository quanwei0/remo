"""SectionedPlaybook in both styles: skeletons, ADD operations, ids, stats, reinforcement, consolidator
reply parsing and persistence (no model)."""
import os
import tempfile
import unittest

from remo import SectionedPlaybook
from remo.memory import find_cited_ids

ADD = lambda section, content: {"type": "ADD", "section": section, "content": content}


class TestSkeleton(unittest.TestCase):
    def test_counts_skeleton(self):
        pb = SectionedPlaybook.from_skeleton("counts")
        self.assertTrue(pb.text.startswith("## STRATEGIES & INSIGHTS\n\n## FORMULAS & CALCULATIONS"))
        self.assertTrue(pb.text.endswith("## OTHERS"))
        self.assertEqual((len(pb), pb.ids(), pb.next_id), (0, [], 1))
        self.assertEqual(pb.render(), pb.text); self.assertEqual(pb.render(10), pb.text[:10])

    def test_plain_skeleton(self):
        pb = SectionedPlaybook.from_skeleton("plain")
        self.assertTrue(pb.text.startswith("## STRATEGIES AND HARD RULES\n\n## APIs TO USE FOR SPECIFIC INFORMATION"))
        self.assertIn("## TROUBLESHOOTING AND PITFALLS:\n\n## OTHERS", pb.text)
        self.assertEqual(len(pb), 0)

    def test_style_checked(self):
        with self.assertRaises(AssertionError):
            SectionedPlaybook("", "other")


class TestAdd(unittest.TestCase):
    def test_counts_known_section_goes_after_the_blank_line(self):
        pb = SectionedPlaybook.from_skeleton("counts")
        self.assertEqual(pb.apply_add_ops([ADD("FORMULAS & CALCULATIONS", "Use average equity.")]), ["calc-00001"])
        self.assertIn("## FORMULAS & CALCULATIONS\n\n[calc-00001] helpful=0 harmful=0 :: Use average equity.\n## CODE SNIPPETS", pb.text)
        self.assertEqual((len(pb), pb.ids(), pb.next_id), (1, ["calc-00001"], 2))

    def test_counts_unknown_section_goes_to_others_and_ids_are_global(self):
        pb = SectionedPlaybook.from_skeleton("counts")
        pb.apply_add_ops([ADD("formulas_and_calculations", "a")])
        self.assertEqual(pb.apply_add_ops([ADD("no such section", "b"), ADD("Common Mistakes To Avoid", "c")]), ["misc-00002", "err-00003"])
        self.assertTrue(pb.text.endswith("## OTHERS\n[misc-00002] helpful=0 harmful=0 :: b"))
        self.assertEqual(pb.ids(), ["calc-00001", "err-00003", "misc-00002"])     # text order, not id order
        self.assertEqual(pb.apply_add_ops([]), []); self.assertEqual(pb.next_id, 4)

    def test_plain_lines_have_no_counters(self):
        pb = SectionedPlaybook.from_skeleton("plain")
        self.assertEqual(pb.apply_add_ops([ADD("STRATEGIES AND HARD RULES", "Check the docs."),
                                           ADD("Troubleshooting and Pitfalls:", "colon ok")]), ["shr-00001", "ts-00002"])
        self.assertIn("## STRATEGIES AND HARD RULES\n\n[shr-00001] Check the docs.\n## APIs", pb.text)
        self.assertIn("[ts-00002] colon ok", pb.text); self.assertEqual(len(pb), 2)

    def test_stats_keys(self):
        c = SectionedPlaybook.from_skeleton("counts"); c.apply_add_ops([ADD("OTHERS", "x")]); c.reinforce("misc-00001")
        self.assertEqual(c.stats(), {"total_bullets": 1, "high_performing": 0, "problematic": 0, "unused": 0,
                                     "by_section": {"OTHERS": {"count": 1, "helpful": 1, "harmful": 0}}})
        p = SectionedPlaybook.from_skeleton("plain"); p.apply_add_ops([ADD("OTHERS", "x"), ADD("OTHERS", "y")])
        self.assertEqual(p.stats(), {"total_bullets": 2, "by_section": {"OTHERS": {"count": 2}}})


class TestReinforce(unittest.TestCase):
    def test_counts_helpful_plus_one(self):
        pb = SectionedPlaybook.from_skeleton("counts"); pb.apply_add_ops([ADD("OTHERS", "x")])
        self.assertTrue(pb.reinforce("misc-00001")); self.assertTrue(pb.reinforce("misc-00001"))
        self.assertIn("[misc-00001] helpful=2 harmful=0 :: x", pb.text)
        self.assertFalse(pb.reinforce("misc-00009")); self.assertEqual(len(pb), 1)

    def test_plain_confirmed_tag(self):
        pb = SectionedPlaybook.from_skeleton("plain"); pb.apply_add_ops([ADD("OTHERS", "x"), ADD("OTHERS", "y")])
        self.assertTrue(pb.reinforce("misc-00001"))
        self.assertIn("[misc-00001] x [confirmed x2]\n[misc-00002] y", pb.text)      # the runs' tag starts at x2
        pb.reinforce("misc-00001")
        self.assertIn("[misc-00001] x [confirmed x3]\n", pb.text); self.assertEqual(pb.ids(), ["misc-00001", "misc-00002"])
        self.assertFalse(pb.reinforce("misc-00003"))


class TestOpsResponse(unittest.TestCase):
    OK = '{"reasoning": "r", "operations": [{"type": "ADD", "section": "OTHERS", "content": "a"}]}'

    def test_valid_shapes(self):
        pb = SectionedPlaybook.from_skeleton("counts")
        self.assertEqual(len(pb.parse_ops_response("Sure:\n```json\n" + self.OK + "\n```")), 1)
        self.assertEqual(len(pb.parse_ops_response("note {x} " + self.OK + " }")), 1)
        self.assertEqual(pb.parse_ops_response('{"reasoning": "r", "operations": []}'), [])

    def test_invalid(self):
        pb = SectionedPlaybook.from_skeleton("counts")
        for bad in ("garbage", "[1]", '{"operations": []}', '{"reasoning": "r", "operations": "x"}',
                    '{"reasoning": "r", "operations": [{"section": "OTHERS", "content": "a"}]}',
                    '{"reasoning": "r", "operations": [{"type": "ADD", "section": "OTHERS"}]}', ""):
            self.assertIsNone(pb.parse_ops_response(bad), bad)

    def test_other_operation_types_per_style(self):
        two = '{"reasoning": "r", "operations": [{"type": "UPDATE", "bullet_id": "x"}, {"type": "ADD", "section": "OTHERS", "content": "a"}]}'
        self.assertEqual(len(SectionedPlaybook.from_skeleton("counts").parse_ops_response(two)), 1)
        self.assertIsNone(SectionedPlaybook.from_skeleton("plain").parse_ops_response(two))

    def test_plain_section_filter(self):
        pb = SectionedPlaybook.from_skeleton("plain")
        ops = pb.parse_ops_response('{"reasoning": "r", "operations": [{"type": "ADD", "section": "Verification Checklist", "content": "v"},'
                                        '{"type": "ADD", "section": "FORMULAS", "content": "dropped"}]}')
        self.assertEqual([o["content"] for o in ops], ["v"])
        self.assertEqual(pb.apply_add_ops(ops), ["vc-00001"])


class TestPersistence(unittest.TestCase):
    def test_roundtrip_is_byte_identical(self):
        text = "## STRATEGIES AND HARD RULES\n[shr-00003] keep me [confirmed x2]\n\n## OTHERS\n"
        p = os.path.join(tempfile.mkdtemp(), "playbook.txt")
        SectionedPlaybook(text, "plain").save(p)
        pb = SectionedPlaybook.load(p, "plain")
        self.assertEqual(pb.text, text); self.assertEqual((pb.next_id, pb.ids()), (4, ["shr-00003"]))
        self.assertEqual(pb.apply_add_ops([ADD("others", "new")]), ["misc-00004"])
        self.assertFalse(pb.text.endswith("\n"))                                    # applying strips, as the runs did


class TestCitedIds(unittest.TestCase):
    def test_order_and_dedup(self):
        self.assertEqual(find_cited_ids("see [calc-00002] and [misc-00001]", "[calc-00002] again", None), ["calc-00002", "misc-00001"])
        self.assertEqual(SectionedPlaybook.find_cited_ids(""), [])


if __name__ == "__main__":
    unittest.main()

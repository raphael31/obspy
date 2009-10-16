# -*- coding: utf-8 -*-

from StringIO import StringIO
from glob import glob
from lxml import etree
from obspy.xseed.blockette.blockette010 import Blockette010
from obspy.xseed.blockette.blockette051 import Blockette051
from obspy.xseed.blockette.blockette054 import Blockette054
from obspy.xseed.parser import Parser, SEEDParserException
from obspy.xseed.utils import compareSEED
import inspect
import os
import unittest


class ParserTestCase(unittest.TestCase):

    def setUp(self):
        # directory where the test files are located
        self.dir = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(self.dir, 'data')

    def tearDown(self):
        pass

    def test_invalidStartHeader(self):
        """
        A SEED Volume must start with a Volume Index Control Header.
        """
        data = "000001S 0510019~~0001000000"
        sp = Parser(strict=True)
        self.assertRaises(SEEDParserException, sp.parseSEED, StringIO(data))

    def test_invalidStartBlockette(self):
        """
        A SEED Volume must start with Blockette 010.
        """
        data = "000001V 0510019~~0001000000"
        sp = Parser(strict=True)
        self.assertRaises(SEEDParserException, sp.parseSEED, StringIO(data))

    def test_blocketteStartsAfterRecord(self):
        """
        '... 058003504 1.00000E+00 0.00000E+0000 000006S*0543864 ... '
        ' 0543864' -> results in Blockette 005
        """
        # create a valid blockette 010 with record length 256
        b010 = "0100018 2.408~~~~~"
        blockette = Blockette010(strict=True)
        blockette.parseSEED(b010)
        self.assertEquals(b010, blockette.getSEED())
        # create a valid blockette 054
        b054 = "0540240A0400300300000009" + ("+1.58748E-03" * 18)
        blockette = Blockette054(strict=True)
        blockette.parseSEED(b054)
        self.assertEquals(b054, blockette.getSEED())
        # combine data
        data = "000001V " + b010 + (' ' * 230)
        data += "000002S " + b054 + (' ' * 8)
        data += "000003S*" + b054 + (' ' * 8)
        # read records
        parser = Parser(strict=True)
        parser.parseSEED(StringIO(data))

    def test_multipleContinuedStationControlHeader(self):
        """
        """
        # create a valid blockette 010 with record length 256
        b010 = "0100018 2.408~~~~~"
        blockette = Blockette010(strict=True)
        blockette.parseSEED(b010)
        self.assertEquals(b010, blockette.getSEED())
        # create a valid blockette 054
        b054 = "0540960A0400300300000039"
        nr = ""
        for i in range(0, 78):
            nr = nr + "+1.000%02dE-03" % i # 960 chars
        blockette = Blockette054(strict=True)
        blockette.parseSEED(b054 + nr)
        self.assertEquals(b054 + nr, blockette.getSEED())
        # create a blockette 052
        b051 = '05100271999,123~~0001000000'
        blockette = Blockette051(strict=True)
        blockette.parseSEED(b051)
        # combine data (each line equals 256 chars)
        data = "000001V " + b010 + (' ' * 230)
        data += "000002S " + b054 + nr[0:224] # 256-8-24 = 224
        data += "000003S*" + nr[224:472] # 256-8 = 248
        data += "000004S*" + nr[472:720]
        data += "000005S*" + nr[720:] + b051 + ' ' * 5 # 5 spaces left
        self.assertEqual(len(data), 256 * 5)
        data += "000006S " + b054 + nr[0:224] # 256-8-24 = 224
        data += "000007S*" + nr[224:472] # 256-8 = 248
        data += "000008S*" + nr[472:720]
        data += "000009S*" + nr[720:] + ' ' * 32 # 32 spaces left
        self.assertEqual(len(data), 256 * 9)
        # read records
        parser = Parser(strict=True)
        parser.parseSEED(StringIO(data))
        # check results
        self.assertEquals(sorted(parser.blockettes.keys()), [10, 51, 54])
        self.assertEquals(len(parser.blockettes[10]), 1)
        self.assertEquals(len(parser.blockettes[51]), 1)
        self.assertEquals(len(parser.blockettes[54]), 2)

    def test_readAndWriteSEED(self):
        """
        Reads all SEED records from the Bavarian network and writes them
        again.
        
        This should not change them.
        
        There are some differences which will be edited before comparison:
        - The written SEED file will always have the version 2.4. BW uses
          version 2.3.
        - There is a missing ~ in Field 9 in Blockette 10. This is a fault
          in the BW files. (see SEED-Manual V2.4 page 38)
        
        The different formating of numbers in the stations blockettes will not
        be changed but 'evened'. Both are valid ways to do it - see SEED-Manual
        chapter 3 for more informations.
        """
        # Get all filenames.
        BW_SEED_files = glob(os.path.join(self.path, u'dataless.seed.BW*'))
        # Loop over all files.
        for file in BW_SEED_files:
            parser = Parser()
            f = open(file, 'r')
            # Original SEED file.
            original_seed = f.read()
            f.seek(0)
            # Parse and write the data.
            parser.parseSEED(f)
            f.close()
            new_seed = parser.getSEED()
            # compare both SEED strings
            compareSEED(original_seed, new_seed)
            del parser
            parser1 = Parser()
            parser2 = Parser()
            original_seed = StringIO(original_seed)
            new_seed = StringIO(new_seed)
            parser1.parseSEED(original_seed)
            parser2.parseSEED(new_seed)
            self.assertEqual(parser1.getSEED(), parser2.getSEED())
            del parser1, parser2

    def test_createReadAssertAndWriteXSEED(self):
        """
        This test takes some SEED files, reads them to a Parser object
        and converts them back to SEED once. This is done to avoid any
        formating issues as seen in test_readAndWriteSEED.
        
        Therefore the reading and writing of SEED files is considered to be
        correct.
        
        Finally the resulting SEED gets converted to XSEED and back to SEED
        and the two SEED strings are then evaluated to be identical.
        
        This tests also checks for XML validity using the a xsd-file.
        """
        # Get all filenames.
        BW_SEED_files = glob(os.path.join(self.path, u'dataless.seed.BW*'))
        # Path to xsd-file.
        xsd_path = os.path.join(self.path, 'xml-seed.xsd')
        # Prepare validator.
        f = open(xsd_path, 'r')
        xmlschema_doc = etree.parse(f)
        f.close()
        xmlschema = etree.XMLSchema(xmlschema_doc)
        # Loop over all files.
        for file in BW_SEED_files:
            parser1 = Parser()
            # Parse the file.
            parser1.parseSEEDFile(file)
            # Convert to SEED once to avoid any issues seen in
            # test_readAndWriteSEED.
            original_seed = parser1.getSEED()
            del parser1
            # Now read the file, parse it, write XSEED, read XSEED and write
            # SEED again. The output should be totally identical.
            parser2 = Parser()
            parser2.parseSEED(StringIO(original_seed))
            xseed_string = parser2.getXSEED()
            del parser2
            # Validate XSEED.
            doc = etree.parse(StringIO(xseed_string))
            self.assertTrue(xmlschema.validate(doc))
            del doc
            parser3 = Parser()
            parser3.parseXSEED(StringIO(xseed_string))
            new_seed = parser3.getSEED()
            self.assertEqual(original_seed, new_seed)
            del parser3, original_seed, new_seed


def suite():
    return unittest.makeSuite(ParserTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')

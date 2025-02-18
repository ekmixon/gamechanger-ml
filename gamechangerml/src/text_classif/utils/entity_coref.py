import logging
import re

import pandas as pd

import gamechangerml.src.text_classif.utils.entity_lookup as el
from gamechangerml.src.text_classif.utils.predict_glob import predict_glob

logger = logging.getLogger(__name__)


class EntityCoref(object):
    def __init__(self):
        """
        This implements a simplistic entity co-reference mechanism geared to
        the structure of many DoD documents.

        When the 'responsibilities' section has been reached, begin looking
        for the word 'shall' in each sentence. If found, see if it contains
        a entity in the lookups. The entity is associated with sentences
        labeled as '1' (modulo details).
        """
        self.RESP = "RESPONSIBILITIES"
        self.SENT = "sentence"
        self.KW = "shall"
        self.KW_RE = re.compile("\\b" + self.KW + "\\b[:,]?")
        self.NA = "Unable to connect Responsibility to Entity"
        self.TOPCLASS = "top_class"
        self.ENT = "entity"

        self.USC_DOT = "U.S.C."
        self.USC = "USC"
        self.USC_RE = "\\b" + self.USC + "\\b"
        self.PL = "P.L."
        self.PL_DOT = "P. L."
        self.PL_RE = "\\b" + self.PL_DOT + "\\b"
        self.EO = "E.O."
        self.EO_DOT = "E. O."
        self.EO_RE = "\\b" + self.EO_DOT + "\\b"

        self.dotted = [self.USC_DOT, self.PL]
        self.subs = [self.USC, self.PL]
        self.sub_back = [self.USC_DOT, self.PL_DOT]
        self.unsub_re = [self.USC_RE, self.PL_RE]

        self.pop_entities = None
        self.contains_entity = el.ContainsEntity()

    def _new_edict(self, value=None):
        value = self.NA or value
        return {self.ENT: value}

    def _re_sub(self, sentence):
        for regex, sub in zip(self.dotted, self.subs):
            sentence = re.sub(regex, sub, sentence)
        return sentence

    def _unsub_df(self, df, regex, sub):
        df[self.SENT] = [re.sub(regex, sub, str(x)) for x in df[self.SENT]]

    def _link_entity(self, output_list, entity_list):
        curr_entity = self.NA
        last_entity = self.NA

        for entry in output_list:
            logger.debug(entry)
            sentence = entry[self.SENT]
            sentence = self._re_sub(sentence)
            new_entry = self._new_edict()
            new_entry.update(entry)

            if entry[self.TOPCLASS] == 0 and self.KW in sentence:
                # current entity is the lhs of the split
                curr_entity = re.split(self.KW_RE, sentence, maxsplit=1)[
                    0
                ].strip()
                # if it's not in the list, set curr_entity to NA
                if not self.contains_entity(curr_entity):
                    curr_entity = self.NA
                else:
                    last_entity = curr_entity
            elif entry[self.TOPCLASS] == 1:
                if curr_entity == self.NA:
                    curr_entity = last_entity
                    # last_entity = curr_entity
                new_entry[self.ENT] = curr_entity
                logger.debug("entity : {}".format(curr_entity))
            entity_list.append(new_entry)

    def _populate_entity(self, output_list):
        entity_list = list()
        for idx, entry in enumerate(output_list):
            e_dict = self._new_edict()
            e_dict.update(entry)
            if e_dict[self.TOPCLASS] == 0 and self.RESP in entry[self.SENT]:
                entity_list.append(e_dict)
                self._link_entity(output_list[idx + 1 :], entity_list)
                return entity_list
            else:
                entity_list.append(e_dict)
        return entity_list

    def make_table(self, model_path, data_path, glob, max_seq_len, batch_size):
        """
        Loop through the documents, predict each piece of text and attach
        an entity.

        The arguments are shown below in `args`.

        A list entry looks like:

            {'top_class': 0,
             'prob': 0.997,
              'src': 'DoDD 5105.21.json',
             'label': 0,
             'sentence': 'Department of...'}

        --> `top_class` is the predicted label

        Returns:
            None

        """
        self.pop_entities = list()
        for output_list, file_name in predict_glob(
            model_path, data_path, glob, max_seq_len, batch_size
        ):
            logger.debug("num input : {:,}".format(len(output_list)))
            self.pop_entities.extend(self._populate_entity(output_list))
            logger.info(
                "processed : {:>4,d}  {}".format(
                    len(self.pop_entities), file_name
                )
            )

    def to_df(self):
        df = pd.DataFrame(self.pop_entities)
        for regex, sub in zip(self.unsub_re, self.sub_back):
            self._unsub_df(df, regex, sub)
        return df

    def to_csv(self, output_csv):
        df = self.to_df()
        df.to_csv(output_csv, index=False)

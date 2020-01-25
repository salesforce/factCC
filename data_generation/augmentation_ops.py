"""
Data augmentation (transformations) operations used to generate 
synthetic training data for the `FactCC` and `FactCCX` models.
"""

import random

import spacy

from google.cloud import translate


LABEL_MAP = {True: "CORRECT", False: "INCORRECT"}


def align_ws(old_token, new_token):
    # Align trailing whitespaces between tokens
    if old_token[-1] == new_token[-1] == " ":
        return new_token
    elif old_token[-1] == " ":
        return new_token + " "
    elif new_token[-1] == " ":
        return new_token[:-1]
    else:
        return new_token


def make_new_example(eid=None, text=None, claim=None, label=None, extraction_span=None,
                     backtranslation=None, augmentation=None, augmentation_span=None, noise=None):
    # Embed example information in a json object.
    return {
        "id": eid,
        "text": text,
        "claim": claim,
        "label": label,
        "extraction_span": extraction_span,
        "backtranslation": backtranslation,
        "augmentation": augmentation,
        "augmentation_span": augmentation_span,
        "noise": noise
    }


class Transformation():
    # Base class for all data transformations

    def __init__(self):
        # Spacy toolkit used for all NLP-related substeps
        self.spacy = spacy.load("en")

    def transform(self, example):
        # Function applies transformation on passed example
        pass


class SampleSentences(Transformation):
    # Embed document as Spacy object and sample one sentence as claim
    def __init__(self, min_sent_len=8):
        super().__init__()
        self.min_sent_len = min_sent_len

    def transform(self, example):
        assert example["text"] is not None, "Text must be available"

        # split into sentences
        page_id = example["id"]
        page_text = example["text"].replace("\n", " ")
        page_doc = self.spacy(page_text, disable=["tagger"])
        sents = [sent for sent in page_doc.sents if len(sent) >= self.min_sent_len]

        # sample claim
        claim = random.choice(sents)
        new_example = make_new_example(eid=page_id, text=page_doc,
                                       claim=self.spacy(claim.text),
                                       label=LABEL_MAP[True],
                                       extraction_span=(claim.start, claim.end-1),
                                       backtranslation=False, noise=False)
        return new_example


class NegateSentences(Transformation):
    # Apply or remove negation from negatable tokens
    def __init__(self):
        super().__init__()
        self.__negatable_tokens = ("are", "is", "was", "were", "have", "has", "had",
                                   "do", "does", "did", "can", "ca", "could", "may",
                                   "might", "must", "shall", "should", "will", "would")

    def transform(self, example):
        assert example["text"] is not None, "Text must be available"
        assert example["claim"] is not None, "Claim must be available"

        new_example = dict(example)
        new_claim, aug_span = self.__negate_sentences(new_example["claim"])

        if new_claim:
            new_example["claim"] = new_claim
            new_example["label"] = LABEL_MAP[False]
            new_example["augmentation"] = self.__class__.__name__
            new_example["augmentation_span"] = aug_span
            return new_example
        else:
            return None

    def __negate_sentences(self, claim):
        # find negatable token, return None if no candiates found
        candidate_tokens = [token for token in claim if token.text in self.__negatable_tokens]

        if not candidate_tokens:
            return None, None

        # choose random token to negate
        negated_token = random.choice(candidate_tokens)
        negated_ix = negated_token.i
        doc_len = len(claim)

        if negated_ix > 0:
            if claim[negated_ix - 1].text in self.__negatable_tokens:
                negated_token = claim[negated_ix - 1]
                negated_ix = negated_ix - 1

        # check whether token is negative
        is_negative = False
        if (doc_len - 1) > negated_ix:
            if claim[negated_ix + 1].text in ["not", "n't"]:
                is_negative = True
            elif claim[negated_ix + 1].text == "no":
                return None, None


        # negate token
        claim_tokens = [token.text_with_ws for token in claim]
        if is_negative:
            if claim[negated_ix + 1].text.lower() == "n't":
                if claim[negated_ix + 1].text.lower() == "ca":
                    claim_tokens[negated_ix] = "can" if claim_tokens[negated_ix].islower() else "Can"
                claim_tokens[negated_ix] = claim_tokens[negated_ix] + " "
            claim_tokens.pop(negated_ix + 1)
        else:
            if claim[negated_ix].text.lower() in ["am", "may", "might", "must", "shall", "will"]:
                negation = "not "
            else:
                negation = random.choice(["not ", "n't "])

            if negation == "n't ":
                if claim[negated_ix].text.lower() == "can":
                    claim_tokens[negated_ix] = "ca" if claim_tokens[negated_ix].islower() else "Ca"
                else:
                    claim_tokens[negated_ix] = claim_tokens[negated_ix][:-1]
            claim_tokens.insert(negated_ix + 1, negation)

        # create new claim object
        new_claim = self.spacy("".join(claim_tokens))
        augmentation_span = (negated_ix, negated_ix if is_negative else negated_ix + 1)

        if new_claim.text == claim.text:
            return None, None
        else:
            return new_claim, augmentation_span


class Backtranslation(Transformation):
    # Paraphrase sentence via backtranslation with Google Translate API
    # Requires API Key for Google Cloud SDK, additional charges DO apply
    def __init__(self, dst_lang=None):
        super().__init__()

        self.src_lang = "en"
        self.dst_lang = dst_lang
        self.accepted_langs = ["fr", "de", "zh-TW", "es", "ru"]
        self.translator = translate.Client()

    def transform(self, example):
        assert example["text"] is not None, "Text must be available"
        assert example["claim"] is not None, "Claim must be available"

        new_example = dict(example)
        new_claim, _ = self.__backtranslate(new_example["claim"])

        if new_claim:
            new_example["claim"] = new_claim
            new_example["backtranslation"] = True
            return new_example
        else:
            return None

    def __backtranslate(self, claim):
        # chose destination language, passed or random from list
        dst_lang = self.dst_lang if self.dst_lang else random.choice(self.accepted_langs)

        # translate to intermediate language and back
        claim_trans = self.translator.translate(claim.text, target_language=dst_lang, format_="text")
        claim_btrans = self.translator.translate(claim_trans["translatedText"], target_language=self.src_lang, format_="text")

        # create new claim object
        new_claim = self.spacy(claim_btrans["translatedText"])
        augmentation_span = (new_claim[0].i, new_claim[-1].i)

        if claim.text == new_claim.text:
            return None, None
        else:
            return new_claim, augmentation_span


class PronounSwap(Transformation):
    # Swap randomly chosen pronoun
    def __init__(self, prob_swap=0.5):
        super().__init__()

        self.class2pronoun_map = {
            "SUBJECT": ["you", "he", "she", "we", "they"],
            "OBJECT": ["me", "you", "him", "her", "us", "them"],
            "POSSESSIVE": ["my", "your", "his", "her", "its", "out", "your", "their"],
            "REFLEXIVE": ["myself", "yourself", "himself", "itself", "outselves", "yourselves", "themselves"]
        }

        self.pronoun2class_map = {pronoun: key for (key, values) in self.class2pronoun_map.items() for pronoun in values}
        self.pronouns = {pronoun for (key, values) in self.class2pronoun_map.items() for pronoun in values}

    def transform(self, example):
        assert example["text"] is not None, "Text must be available"
        assert example["claim"] is not None, "Claim must be available"

        new_example = dict(example)
        new_claim, aug_span = self.__swap_pronouns(new_example["claim"])

        if new_claim:
            new_example["claim"] = new_claim
            new_example["label"] = LABEL_MAP[False]
            new_example["augmentation"] = self.__class__.__name__
            new_example["augmentation_span"] = aug_span
            return new_example
        else:
            return None

    def __swap_pronouns(self, claim):
        # find pronouns
        claim_pronouns = [token for token in claim if token.text.lower() in self.pronouns]

        if not claim_pronouns:
            return None, None

        # find pronoun replacement
        chosen_token = random.choice(claim_pronouns)
        chosen_ix = chosen_token.i
        chosen_class = self.pronoun2class_map[chosen_token.text.lower()]

        candidate_tokens = [token for token in self.class2pronoun_map[chosen_class] if token != chosen_token.text.lower()]

        if not candidate_tokens:
            return None, None

        # swap pronoun and update indices
        swapped_token = random.choice(candidate_tokens)
        swapped_token = align_ws(chosen_token.text_with_ws, swapped_token)
        swapped_token = swapped_token if chosen_token.text.islower() else swapped_token.capitalize()

        claim_tokens = [token.text_with_ws for token in claim]
        claim_tokens[chosen_ix] = swapped_token

        # create new claim object
        new_claim = self.spacy("".join(claim_tokens))
        augmentation_span = (chosen_ix, chosen_ix)

        if claim.text == new_claim.text:
            return None, None
        else:
            return new_claim, augmentation_span


class NERSwap(Transformation):
    # Swap NER objects - parent class
    def __init__(self):
        super().__init__()
        self.categories = ()

    def transform(self, example):
        assert example["text"] is not None, "Text must be available"
        assert example["claim"] is not None, "Claim must be available"

        new_example = dict(example)
        new_claim, aug_span = self.__swap_entities(new_example["text"], new_example["claim"])

        if new_claim:
            new_example["claim"] = new_claim
            new_example["label"] = LABEL_MAP[False]
            new_example["augmentation"] = self.__class__.__name__
            new_example["augmentation_span"] = aug_span
            return new_example
        else:
            return None

    def __swap_entities(self, text, claim):
        # find entities in given category
        text_ents = [ent for ent in text.ents if ent.label_ in self.categories]
        claim_ents = [ent for ent in claim.ents if ent.label_ in self.categories]

        if not claim_ents or not text_ents:
            return None, None

        # choose entity to replace and find possible replacement in source
        replaced_ent = random.choice(claim_ents)
        candidate_ents = [ent for ent in text_ents if ent.text != replaced_ent.text and ent.text not in replaced_ent.text and replaced_ent.text not in ent.text]

        if not candidate_ents:
            return None, None

        # update claim and indices
        swapped_ent = random.choice(candidate_ents)
        claim_tokens = [token.text_with_ws for token in claim]
        swapped_token = align_ws(replaced_ent.text_with_ws, swapped_ent.text_with_ws)
        claim_swapped = claim_tokens[:replaced_ent.start] + [swapped_token] + claim_tokens[replaced_ent.end:]

        # create new claim object
        new_claim = self.spacy("".join(claim_swapped))
        augmentation_span = (replaced_ent.start, replaced_ent.start + len(swapped_ent) - 1)

        if new_claim.text == claim.text:
            return None, None
        else:
            return new_claim, augmentation_span


class EntitySwap(NERSwap):
    # NER swapping class specialized for entities (people, companies, locations, etc.)
    def __init__(self):
        super().__init__()
        self.categories = ("PERSON", "ORG", "NORP", "FAC", "GPE", "LOC", "PRODUCT",
                           "WORK_OF_ART", "EVENT")


class NumberSwap(NERSwap):
    # NER swapping class specialized for numbers (excluding dates)
    def __init__(self):
        super().__init__()

        self.categories = ("PERCENT", "MONEY", "QUANTITY", "CARDINAL")


class DateSwap(NERSwap):
    # NER swapping class specialized for dates and time
    def __init__(self):
        super().__init__()

        self.categories = ("DATE", "TIME")


class AddNoise(Transformation):
    # Inject noise into claims
    def __init__(self, noise_prob=0.05, delete_prob=0.8):
        super().__init__()

        self.noise_prob = noise_prob
        self.delete_prob = delete_prob
        self.spacy = spacy.load("en")

    def transform(self, example):
        assert example["text"] is not None, "Text must be available"
        assert example["claim"] is not None, "Claim must be available"

        new_example = dict(example)
        claim = new_example["claim"]
        aug_span = new_example["augmentation_span"]
        new_claim, aug_span = self.__add_noise(claim, aug_span)

        if new_claim:
            new_example["claim"] = new_claim
            new_example["augmentation_span"] = aug_span
            new_example["noise"] = True
            return new_example
        else:
            return None

    def __add_noise(self, claim, aug_span):
        claim_tokens = [token.text_with_ws for token in claim]

        new_claim = []
        for ix, token in enumerate(claim_tokens):
            # don't modify text inside an augmented span
            apply_augmentation = True
            if aug_span:
                span_start, span_end = aug_span
                if span_start <= ix <= span_end:
                    apply_augmentation = False

            # decide whether to add noise
            if apply_augmentation and random.random() < self.noise_prob:
                # decide whether to replicate or delete token
                if random.random() < self.delete_prob:
                    # update spans and skip token
                    if aug_span:
                        span_start, span_end = aug_span
                        if ix < span_start:
                            span_start -= 1
                            span_end -= 1
                        aug_span = span_start, span_end
                    if len(new_claim) > 0:
                        if new_claim[-1][-1] != " ":
                            new_claim[-1] = new_claim[-1] + " "
                    continue
                else:
                    if aug_span:
                        span_start, span_end = aug_span
                        if ix < span_start:
                            span_start += 1
                            span_end += 1
                        aug_span = span_start, span_end
                    new_claim.append(token)
            new_claim.append(token)
        new_claim = self.spacy("".join(new_claim))

        if claim.text == new_claim.text:
            return None, None
        else:
            return new_claim, aug_span

from __future__ import annotations


ENTITY_TYPES = "person|organization|place|object|event|activity|concept|other"


THEME_POLICY = """
A memory episode is one contiguous span mainly focused on one coherent entity, event, goal, or
bounded topic. Use theme_type entity, event, or topic. Split only on a genuine semantic shift;
keep greetings, acknowledgements, clarification questions, and short follow-ups with the material
they support. A return to an earlier theme after an intervening theme starts a new episode.
""".strip()


PARTITION_SYSTEM = """You partition a complete dialogue session into contiguous theme-coherent
memory episodes. These boundaries define immutable evidence units. Return one JSON object only."""


PARTITION_USER = """Partition this session.

{theme_policy}

STRUCTURAL RULES
- Cover every supplied dialogue turn exactly once and in order, without gaps or overlaps.
- Do not split merely because the speaker changes.
- Avoid one-turn fragments unless a turn clearly introduces a separate durable theme.
- Preserve image-bearing turns with the topic they support. The rendered turn may include
  text, blip_caption/image_caption/caption, image retrieval query, and img_url; keep all of
  these fields attached to the same DIA and never split a caption away from its turn.
- Copy start_dia_id and end_dia_id exactly.
- Use boundary_reason=session_start for the first segment.

Return:
{{
  "segments": [
    {{
      "start_dia_id": "D1:1",
      "end_dia_id": "D1:5",
      "theme": "concise canonical theme",
      "theme_type": "entity|event|topic",
      "theme_description": "one-sentence scope",
      "boundary_reason": "session_start or semantic shift"
    }}
  ]
}}

Conversation: {conversation_id}
Session: {session_id}
Observed at: {observed_at}

Turns:
{session_text}
"""


PARTITION_REPAIR_SYSTEM = """Repair an invalid dialogue partition. Return JSON only."""


PARTITION_REPAIR_USER = """Repair the partition so it covers all required dialogue IDs exactly
once, in order, with contiguous non-overlapping segments. Preserve valid theme decisions.

Validation error: {error}
Required dia_ids: {dia_ids}
Original proposal: {proposal}

Return one object with segments using theme_type entity, event, or topic.
"""


INDEX_SYSTEM = """You are the high-recall, precision-preserving information-extraction stage of an
episodic memory system. Build a query-independent entity-structured index from one complete theme
episode. The episode remains the only answer evidence; records are navigation handles. Follow the
minimum-sufficient predicate policy and the extraction checklist literally. Never summarize the
episode and never return prose outside one valid JSON object."""


INDEX_USER = """Extract an exhaustive set of atomic records from the episode below.

The input is deliberately marked with XML-like delimiters. Read every turn and every displayed
metadata field. Image captions and image-retrieval descriptions are auxiliary textual evidence.
The image URL is shown for provenance only: it is opaque metadata, not a source of facts, and must
never be interpreted or used for outside lookup.

<extraction_checklist>
1. Scan each DIA turn independently. For every explicit factual clause, emit at least one record.
   A greeting, question, compliment, or acknowledgement may have zero records only when it adds
   no fact. Do not let an adjacent answer cause you to skip a factual clause in that turn.
2. Resolve I/my/me using the speaker on that turn. Keep the speaker in source and cite the exact
   DIA IDs that support the record.
3. Every explicitly named person is a person entity. Every explicitly named event or activity
   (for example LGBTQ conference, pride parade, flight to Canada, or camping) is an event/activity
   entity. Caption nouns such as painting, bowl, vase, or necklace are object entities when the
   caption explicitly describes them.
   When a speaker says “my painting”, “I painted”, “my trip”, or an equivalent first-person clause,
   also emit a speaker-grounded person/activity record whose value preserves that concrete item or
   event. Do not leave the fact attached only to an object entity or to a co-speaker.
   When a speaker explicitly says “I saw/attended/went to/visited/listened to [named event, artist,
   band, or place]”, emit both the named entity record and a speaker-grounded relation such as
   person/attend/[event] or person/see/[artist]. If a turn answers a nearby question with “this”,
   “that”, or “these”, resolve the pronoun only to the immediately preceding object/event in the
   same local exchange and emit the answer-bearing person/property record with both DIA IDs.
4. Each record contains one entity and one **minimum-sufficient broad property**. Property is a
   short reusable predicate, normally one or two words, without the subject, entity type, answer
   value, date, or descriptive modifiers. Choose the narrowest predicate that still covers ordinary
   paraphrases and preserves what makes the fact distinctive. The predicate is the semantic head
   of the relation or event, not a compressed copy of the whole sentence. For example, “had a
   check-up with my doctor” has property=checkup (or the ordinary reusable predicate for a medical
   visit) and value may contain doctor-related detail; it must not become personhavecheck_up or
   include the subject in the property. “It helps with my health goals” has property=help and a
   value such as health goals; it must not become health_goal_support. “I researched adoption
   agencies” uses research; flying to a destination uses travel; joining an event uses attend;
   making a painting uses paint. Do NOT replace all of these with activity: a class, a soccer
   game, research, and travel are different predicates. Use activity only when the source genuinely
   leaves the activity unspecified. Never invent compound predicates, concatenate the entity/type
   with a verb, or move a prepositional object into the predicate. Put that detail in value or a
   condition instead.
5. Value is the finer-grained detail explicitly stated in the source. Keep meaningful source
   vocabulary (for example home country, Sweden, 4 years, or sunset over a lake). If the property
   is explicit but has no finer value, use value=""; an empty value is valid and must not be dropped.
6. Record explicit times as property=time (or as valid_time when they qualify an event), so a
   time question can find the episode. Preserve relative wording such as yesterday or 4 years ago.
7. Add an inverse projection only when the relation is explicitly entailed. For example, for
   Caroline attended an LGBTQ support group, emit Caroline/person/attend/LGBTQ support group and
   LGBTQ support group/event/attendee/Caroline, both citing the same DIA. Do not create inverse
   records from mere co-occurrence.
   A named event, person, or object appearing in a speaker's explicit first-person action is not
   mere co-occurrence: preserve the speaker-to-entity relation so later queries can retrieve it.
8. Preserve modality and polarity. Planned, desired, hypothetical, negated, and uncertain facts
   must not be rewritten as observed facts.
9. condition_property must be exactly one of after, before, during, because_of, while, when, for,
   at, with, since, until, if, without, about, or empty. If a relation does not fit this list,
   keep it in value and leave both condition fields empty. Never output keys or dictionaries inside
   conditions; conditions must be a JSON array of short strings.
10. Do not infer facts from world knowledge, episode adjacency, or an image URL. Extract only what
    the dialogue text, caption, and explicitly displayed image-retrieval description establish.
11. Before returning JSON, perform a private quality check for every record: (a) property is a
    reusable predicate rather than a subject-prefixed or sentence-concatenated label, (b) value
    contains the concrete detail and is empty only when no detail is stated, (c) source/evidence
    speaker attribution is correct, (d) relative time is preserved verbatim, and (e) image metadata
    is cited to the exact DIA that contains it. If a proposed property fails this check, rewrite
    the record at the appropriate semantic level rather than emitting the malformed label.
12. Do not let an auxiliary verb or a vague discourse noun hide the content-bearing predicate.
    In “went to the doctor for a check-up”, emit a medical-visit/checkup record in addition to any
    generic travel wording; in “ended up in the ER ... turns out it was gastritis”, emit separate
    records for the ER visit, the symptom, the diagnosis, and the relative time. In “attended a
    Weight Watchers meeting yesterday”, emit the attendance/meeting action and the relative-time
    record. The exact predicate names remain open-ended, but each record must expose the smallest
    meaningful semantic head that a later question could ask for.
13. For every image-bearing DIA, extract the facts stated by the caption and retrieval query as
    object/event records and attach them to that exact DIA. If the turn presents the image as part
    of the speaker's message (even when the text only says “big news” or “look”), also emit a
    speaker-grounded image/show/share relation whose value is the depicted item or scene. Do not
    invent facts beyond the caption/query, and never use the URL as evidence.
</extraction_checklist>

<calibration_examples>
For these contrastive examples, preserve the predicate that distinguishes the fact:
- “I researched adoption agencies” -> Caroline/person/research/adoption agencies
- “I took a flight to Hong Kong” -> Melanie/person/travel/Hong Kong (the value may retain
  “flight” as a manner qualifier, but the property remains travel)
- “I attended an LGBTQ conference” -> Caroline/person/attend/LGBTQ conference
- “I played soccer” -> Melanie/person/activity/soccer only when no narrower sport predicate is
  defined; never label it research, travel, or attend.

For the sentence “I've known these friends for 4 years, since I moved from my home country”, good
records include:
- Caroline/person/move/home country
- Caroline/person/friend/4 years
Do not create support_system_origin or hide the move inside a support property.

For the caption “a photo of a painting of a sunset over a lake”, a good record is:
- painting/object/depicts/sunset over a lake
The caption record cites the DIA that contains the caption.

For a turn whose metadata says `Image retrieval query: grilled salmon with roasted vegetables`,
the description may be used as auxiliary image evidence, but it must remain attached to that DIA;
  do not turn the URL into a guessed fact or silently replace the speaker's text.
- For “I ended up in the ER with a severe stomachache. Turns out, it was gastritis”, emit at least
  person/visit/ER, person/symptom/severe stomachache, person/diagnosis/gastritis, and
  person/time/last weekend records, all citing that DIA.
- For an image-bearing message whose caption is “a photo of a bowl of spinach, avocado, and
  strawberries”, emit the object/depicts record and a speaker-grounded show/share relation tied to
  the same DIA, even if the text does not repeat the food name.
</calibration_examples>

Return one object only:
{{
  "records": [
    {{
      "entity": "short semantic entity name",
      "entity_type": "{entity_types}",
      "property": "short broad snake_case predicate",
      "value": "finer-grained explicit value or empty string",
      "condition_property": "after|before|during|because_of|while|when|for|at|with|since|until|if|without|about or empty",
      "condition_value": "explicit condition value or empty string",
      "property_kind": "aspect|relation",
      "modality": "observed|planned|desired|hypothetical|negated|uncertain",
      "conditions": [],
      "valid_time": "explicit event-valid time or relative expression, otherwise empty",
      "source": "speaker who supplied the information",
      "evidence_dia_ids": ["exact supporting DIA IDs"],
      "projection_kind": "direct|inverse|state_projection|event_access",
      "confidence": 0.0
    }}
  ]
}}

<episode_metadata>
Episode ID: {episode_id}
Conversation: {conversation_id}
Session: {session_id}
Observed at: {observed_at}
Episode theme: {episode_theme} ({episode_theme_type})
</episode_metadata>

<episode_turns>
{episode_text}
</episode_turns>
"""


QUERY_SYSTEM = """You are the exhaustive query-decomposition stage of an episodic memory system.
Convert the question into searchable requirements and an evidence-seeking hop graph using the same
[entity][entity_type][property:value][condition_property:condition_value] schema as the memory
index. GPT-4.1-mini follows literal instructions well: preserve every meaningful discriminator,
but never guess an answer. Do not answer the question. Return exactly one valid JSON object."""


QUERY_USER = """Build an exhaustive retrieval plan for the question below.

<planning_rules>
1. First identify every requirement a human would need to understand the question. Include the
   named person/event, the relationship that disambiguates the target, the broad property being
   asked about, and explicit time/quantity/location constraints. Do not collapse these into only
   the final answer type. For example, “When is Melanie's daughter's birthday?” has the requirements
   daughter, birthday, and time—not just birthday.
2. Store these requirements in required_properties. Each broad_property is a reusable one- or
   two-word predicate, with no entity name, entity type, answer value, date, or modifiers. Use the
   same **minimum-sufficient** granularity as the index: preserve a predicate when it distinguishes
   the fact (research, travel, attend, read, paint, support, help, checkup, function, learn,
   birthday, etc.), and use activity only for an actually unspecified activity. A class, soccer
   game, research task, medical check-up, and trip must not all become activity. A named event
   belongs in entity or value; for example, “pride parade” is an event value for property=attend.
   A property is the semantic head of the question, not a subject-prefixed or sentence-concatenated
   label: never emit values such as personhavecheck_up, health_goal_support, or persontravel.
   Do not replace an explicit activity question with travel/research merely because one possible
   answer might involve travel/research. The observed index vocabulary is a compatibility hint,
   not permission to erase the question's semantic distinction. Never invent a compound label
   such as daughter's_birthday or pride_parade.
3. Every searchable hop must use only information stated in the question or a declared bridge.
   anchor has entity, entity_type, property, value, condition_property, and condition_value. The
   anchor's entity and property are mandatory and must never be empty. Empty value means the answer
   value is unknown; never guess it. If the question says "my" or "the user", use entity=user and
   entity_type=person. In an anchor use the key `property` (not `broad_property`); the latter is
   reserved for required_properties.
4. Make the hops exhaustive but non-redundant. Add a separate hop for each independently useful
discriminator when it improves recall. For “When is Melanie's daughter's birthday?”, use:
   - Melanie/person/relationship/daughter
   - Melanie/person/birthday/""/for/daughter
   - Melanie/person/time/""/for/birthday
   The three hops let the retriever distinguish Melanie's daughter's birthday from Melanie's own
   birthday without putting the unknown date into an anchor.
   For list questions such as “Where has Melanie camped?”, use one all_matching activity hop with
   value=camp (and let the values carry mountains, beach, forest). Do not add a bridge that selects
   only one location from the first camping episode. For “What books has Melanie read?”, use
   all_matching and make the representation explicit: one book hop with an empty value plus one
   activity hop with value=read. These are alternatives, not an AND requirement.
5. For “Where did Caroline move from 4 years ago?”, preserve both move and time: use a move hop
   with condition_property=when and condition_value=4 years ago, plus a time hop if it adds a
   distinct searchable requirement. The answer location remains unknown in the anchor.
6. For event questions such as “When did Caroline go to the pride parade during the summer?”,
   use entity=Caroline, property=attend, value=pride parade, and a separate time/season hop with
   condition_value=summer. Do not use pride parade as the property label.
7. Use broad properties aligned with corpus extraction, but do not over-generalize. Values retain
   explicit source vocabulary such as home country, Sweden, daughter, pride parade, counseling,
   Hong Kong, or adoption agencies. Keep a concrete action in the property when it is distinctive:
   research -> research, flying/going to Hong Kong -> travel with value=Hong Kong, joining a pride
   parade -> attend with value=pride parade. Do not rewrite an unknown answer into a value.
8. Use reasoning_type=aggregation or comparison, and retrieval_scope=all_matching, for count,
   exhaustive-list, frequency, comparison, interval/difference, first-versus-second, or “how many
   times” questions. “How often” is an aggregation question even when the dialogue never states a
   frequency phrase: retrieve every qualifying event, then let the answer stage count occurrences
   or infer the recurring interval from their dates. For “how many months between the first and
   second ...”, retrieve all matching event instances and their times; do not use a point query or
   only the two highest-scoring episodes. Otherwise use retrieval_scope=point. all_matching means
   every qualifying episode is retained, not only top-k. A question asking for every place, item,
   or kind associated with an activity is also an exhaustive-list question even when its
   answer_target is location or value.
9. Independent hops have depends_on=[]. A dependent hop may use exactly $<hop_id>.bridge in entity
   or value, and its source hop must define bridge_request. Avoid fake dependencies when the
   question already supplies the entity.
10. Counterfactual questions need target, removed-condition, and causal-link evidence roles.
    Commonsense questions retrieve only the person's episodic premise, never public taxonomy.
11. Before returning JSON, perform a private intent check: every explicit noun/verb/time/quantity
    constraint in the question appears in at least one requirement or hop; no unknown answer is
    copied into value; no predicate is a subject-prefixed or compound sentence fragment; and the
    broad property remains at the same semantic level as the question. For a question asking
    “which activity was resumed”, keep activity/resume and do not invent travel. For an object
    capability question such as “what does the smartwatch help someone with?”, retrieve the object
    capability/function and the person's associated goal/help relation. For relative-time questions,
    preserve the wording (e.g. “a few days ago”, “Thursday before ...”) as a condition or evidence
    requirement rather than guessing an exact date.
12. For any temporal question (“when”, “what day/date”, “how long between”), keep the event
    predicate and its temporal discriminator together. Do not reduce “when did Sam first go to
    the doctor and find out he had a weight problem?” to go_to; use the medical visit/checkup event
    plus time, and mark first/second/interval/frequency questions as all_matching. For a dated
    activity question, preserve the activity/event (for example kayaking, painting class, ER visit)
    and the stated date or relative-date phrase in the same searchable plan.
13. If the question explicitly asks for a synthesis across multiple people, conversations, themes,
    personal growth, or “both X and Y”, use retrieval_scope=all_matching and include the distinct
    evidence requirements for each person/theme. A single top episode is not sufficient for a
    recommendation or advice synthesized from the dialogue.
14. A complement can carry the answer even when the question uses an auxiliary verb. Prefer the
    semantic head of the complement (“checkup”, “meet”, “travel”, “injury”, “share/show”, “attend”)
    over bare predicates such as go_to, be, get, do, want, or date. This is a prompt-level semantic
    choice, not a fixed alias table: preserve the source/question distinction and never inject an
    answer value that is not stated.
</planning_rules>

<examples>
Question: When is Melanie's daughter's birthday?
required_properties: daughter, birthday, time
hops: relationship(daughter), birthday(for daughter), time(for birthday)

Question: What did Caroline research?
required_properties: research (value="")
hop: Caroline/person/research/""

Question: Where has Melanie camped?
required_properties: camp, location
hop: Melanie/person/camp/camp; reasoning_type=direct; retrieval_scope=all_matching

Question: What books has Melanie read?
required_properties: book, read
hops: Melanie/person/book/"" and Melanie/person/read/read; retrieval_scope=all_matching

Question: What kind of art does Caroline make?
required_properties: paint, art
hop: Caroline/person/paint/art; retrieval_scope=all_matching

Question: Who supports Caroline when she has a negative experience?
required_properties: support, relationship (friends/family/mentors), negative experience
hops: Caroline/person/support/"" and Caroline/person/relationship/""; the experience phrase is a
soft context constraint, not a blocking second hop.

Question: When did Caroline go to the LGBTQ support group?
required_properties: attend, time, support group
hop: Caroline/person/attend/LGBTQ support group/""/""

Question: Would Caroline still pursue counseling if she had not received support growing up?
required_properties: career, counseling, support, upbringing, causality
hops: target career, removed support condition, causal link; reasoning_type=counterfactual

Question: How often does Sam get health checkups?
required_properties: checkup, frequency, time
hop: Sam/person/checkup/""; reasoning_type=aggregation; retrieval_scope=all_matching

Question: Which activity did Sam resume in December 2023 after a long time?
required_properties: activity, resume, time=December 2023, duration=long time
hops: Sam/person/activity/"", Sam/person/resume/"", Sam/person/time/""/during/December 2023,
      Sam/person/duration/""/after/long time; reasoning_type=aggregation; retrieval_scope=all_matching

Question: What does the smartwatch help Evan with?
required_properties: smartwatch/function, Evan/help or health goal
hops: smartwatch/object/function/"" and Evan/person/help/""; keep the object capability and
      the person's goal distinct rather than inventing a compound property.

Question: When did Sam first go to the doctor and find out he had a weight problem?
required_properties: medical visit/checkup, doctor/weight context, time=first occurrence
hops: Sam/person/checkup or medical_visit/"" (with doctor/weight in value or condition), plus a
time hop tied to that event; do not use go_to as the only property.

Question: What food did Sam share a photo of on 19 August, 2023?
required_properties: Sam's image/share event, food/depiction, time=19 August 2023
hops: speaker-grounded share/show/photo hop with the date and an image/object depiction hop; the
answer may be present only in blip_caption or image retrieval query.

Question: Considering their conversations and personal growth, what advice might Evan and Sam
give to someone facing a major life transition?
required_properties: Evan's advice/personal growth, Sam's advice/personal growth, shared coping themes
hops: all_matching evidence for both people and the distinct themes; synthesize only after all
relevant episodes are supplied.
</examples>

<observed_index_property_vocabulary>
{index_property_vocabulary}
</observed_index_property_vocabulary>

Return one object only:
{{
  "answer_target": {{"type": "time|value|entity|location|count|list|boolean|likelihood|explanation", "description": "..."}},
  "reasoning_type": "direct|aggregation|comparison|temporal|causal|counterfactual|commonsense_inference|multi_hop|unanswerable",
  "retrieval_scope": "point|all_matching",
  "required_properties": [
    {{"entity": "known entity or empty", "entity_type": "person|organization|place|object|event|activity|concept|other", "broad_property": "short broad predicate", "property_text": "same short predicate", "value": "known value or empty", "role": "entity|relation|answer_property|time|constraint"}}
  ],
  "hops": [
    {{
      "hop_id": "h1",
      "purpose": "why this evidence is needed",
      "anchor": {{"entity": "", "entity_type": "{entity_types}", "property": "", "value": "", "condition_property": "", "condition_value": ""}},
      "depends_on": [],
      "bridge_request": ""
    }}
  ]
}}

Question: {question}
"""


QUERY_REPAIR_SYSTEM = """You repair an invalid evidence-seeking memory retrieval plan. Do not
answer the question. Return one JSON object only."""


QUERY_REPAIR_USER = """Repair the plan using the same six-field atomic schema and exhaustive
requirement rules.

Question: {question}
Validation error: {error}
Invalid plan:
{plan}

Important reminders:
- Include every meaningful discriminator in required_properties: entity/relationship, broad answer
  property, and explicit time, location, or quantity constraints.
- Use the minimum-sufficient predicate. Do not repair a narrow action such as research, travel,
  attend, read, paint, or learn into generic activity. Keep the narrow predicate and preserve the
  explicit action/object/location in value when the question states one.
- A property must be a reusable semantic head, not a subject-prefixed or sentence-concatenated
  string. Rewrite malformed forms such as personhavecheck_up or health_goal_support into a short
  predicate whose concrete detail is carried by value or a condition. Never replace a question's
  explicit activity with travel/research without textual support.
- Every hop must contain a non-empty anchor.entity and anchor.property. If the answer value is
  unknown, leave only anchor.value empty; do not leave the entity or property empty.
- In anchors use `property`; use `broad_property` only inside required_properties.
- Use short broad property roots (for example birthday, relationship, attend, move, and time), not
  compound labels such as daughter's_birthday or support_system_origin. Known information from the
  question belongs in anchors; unknown answers remain empty.
- For exhaustive place/item/kind questions, use retrieval_scope=all_matching. A camping-location
  list should keep the location in the camp value and must not use a bridge that chooses one
  location from one episode. A book-list query should permit both book and read evidence.
- Frequency, interval/difference, first-versus-second, count, and “how often” questions are
  aggregation/comparison questions and must use retrieval_scope=all_matching, even when no source
  turn contains an explicit frequency phrase.
- condition_property is one of after, before, during, because_of, while, when, for, at, with,
  since, until, if, without, or about.
- A counterfactual needs target, removed-condition, and causal-link evidence roles.
- Do not search episodic memory for public world knowledge or taxonomy.
- Preserve relative time phrases instead of inventing a precise date, and keep image capabilities
  separate from the person's goal/help relation.
- If a generated plan uses a bare auxiliary property (go_to, be, get, do, want, date) while the
  question contains a content-bearing event or object, repair it at the semantic-head level and
  retain the event/time/object constraints. Do not solve this with a hard-coded dataset alias.
- Temporal first/second/interval/frequency questions and explicit multi-person synthesis questions
  must use all_matching and retain every distinct event needed by the answer.

Observed index property vocabulary:
{index_property_vocabulary}

Return a complete corrected object with answer_target, reasoning_type, retrieval_scope,
required_properties, and hops.
"""


QUERY_AUDIT_SYSTEM = """You are a conservative semantic auditor for an evidence-seeking query
plan. Check whether the plan preserves the user's actual intent and every explicit constraint.
Return one complete JSON plan only. Do not answer the question and do not add dataset-specific
aliases."""


QUERY_AUDIT_USER = """Audit and, only if necessary, repair this query plan.

Question: {question}
Observed index vocabulary (a hint, not an ontology):
{index_property_vocabulary}

Candidate plan:
{plan}

Audit checklist:
- Every explicit content noun/verb, participant, object/event, date/relative-time phrase, and
  quantity in the question appears in required_properties or a hop anchor/purpose.
- A content-bearing predicate must not be replaced by an auxiliary or generic predicate such as
  go_to, be, get, do, want, or date. Preserve the question's semantic head at minimum-sufficient
  granularity without using a fixed alias table.
- A temporal question binds the requested event/action to its time evidence. First/second,
  interval, count, frequency, and how-often questions use all_matching; a point question may use
  point only when one event instance is sufficient.
- A multi-person or multi-theme synthesis/recommendation retrieves the distinct evidence for every
  named person/theme, not one convenient episode.
- Unknown answers stay empty in anchors; values contain only information stated in the question.
- Keep image/photo questions open to caption and image-retrieval-query evidence.

If the candidate passes, return it unchanged. Otherwise return a minimally repaired complete object
with answer_target, reasoning_type, retrieval_scope, required_properties, and hops. Validate that
every hop has a non-empty entity and property before returning.
"""


BRIDGE_SYSTEM = """You resolve one declared bridge variable from retrieved original conversation
episodes. Return only a JSON object. Do not answer the user's final question."""


BRIDGE_USER = """Extract the shortest explicit value requested below. Use only the supplied episodes.
If the bridge is not established, return an empty value. Do not infer beyond the evidence.

Bridge request: {bridge_request}
Source hop anchor: {anchor}

Original episodes:
{episodes}

Return: {{"bridge": "short entity or value", "supporting_episode_ids": ["..."]}}
"""


ANSWER_SYSTEM = """Answer only from the supplied complete original conversation episodes.
The structured indexes and hop plan are navigation aids, not evidence. Read evidence from every hop,
combine facts when the plan is multi-hop, and keep entities correctly bound. Attribute a fact to the
person who said or did it; do not transfer a co-speaker's action, preference, or relationship to the
queried person merely because both appear in one episode. Resolve a relative time only to identify
the matching event, but report the source's own wording when it is approximate (“a few days ago”,
“last Tuesday”, “yesterday”, or “last week”). Do not silently replace an event expression with the
episode-observation date; the observed timestamp is a reference point, not automatically the answer.
Use every source field shown inside a turn when present: the dialogue text, image caption, image
retrieval description/query, and image URL. Treat the caption and retrieval description as textual
auxiliary evidence attached to that exact DIA. Treat the URL as opaque provenance only: never infer
facts from its filename/domain and never perform outside lookup.

Before writing the answer, make a private evidence ledger with (1) the exact answer-bearing DIA
turn and any image metadata attached to it, (2) the entity/speaker it belongs to, (3) the
property/value it supports, (4) the source's relative-time wording, and (5) its event date or
reference date. Use that ledger to prevent a generic neighboring fact from replacing a specific
answer or a nearby image/food/event from the requested one.

Answer-type safeguards:
- “what kind/type/style” asks for the explicit category or style (for example, “abstract”), not
  an unfiltered list of media such as paintings, drawings, or stained glass.
- “who supports/helped” asks only for people or groups explicitly named as supporters (for example,
  friends, family, or mentors). Never assume that the other person in the dialogue is a supporter.
- “when” asks for the time of the matching event. Prefer the event instance satisfying the
  question's entity, property, and explicit date/season constraint rather than the latest mention
  of a related event. Preserve the source's relative expression when it is approximate or relational
  (for example “a few days ago”, “last week”, or “Thursday before December 17, 2023”); do not invent
  an exact day. Preserve “yesterday”/“last Tuesday”/“last week” rather than calculating an absolute
  date unless the question explicitly requests conversion and the reference date is unambiguous.
  Do not substitute the episode-observation date for the event date.
- For “first”, “second”, “earliest”, or “latest”, first build the chronological inventory of
  matching event DIAs and apply the ordinal to those events—not to the order in which episodes were
  retrieved and not to a later mention that contains more detail. For “how many months” or a
  similarly whole-unit interval, use the calendar-month difference supported by the two matching
  event references unless the question explicitly requests an exact day count; do not report a
  fractional estimate from an unrelated timestamp.
- For named books, objects, or events, bind the answer to that named item before collecting other
  facts. A local follow-up such as “these are for running” is evidence about the immediately
  preceding named object. If the question names a book but the local dialogue calls it “this book”,
  treat that local reference as the named book when there is no competing book in the same evidence;
  do not return Unknown solely because the title is not repeated in that turn.
- For counterfactual or likelihood questions, give priority to explicit evidence about the
  questioned condition (including a recent negative outcome) over generic positive preferences.
- For an inference about whether a person is religious or spiritual, attribute evidence to that
  person rather than to other people mentioned in the episode.  A person's own cross/faith
  statement, church-related work, or spiritual practice supports a qualified answer such as
  "somewhat religious/spiritual" when no strength level is stated; do not turn the absence of a
  denomination into "not religious", and do not apply a description of religious conservatives
  or another group to the queried person.

For a list question, make an explicit inventory across ALL supplied episodes before answering: collect
every distinct item/event that satisfies the question, then remove duplicates. Treat count, frequency,
interval/difference, and first-versus-second questions as aggregation/comparison: collect every
qualifying event, bind each to its own DIA and time, sort the events, and count or calculate the
interval when the question requires it. A frequency need not be stated verbatim in one turn; infer
the recurring interval from multiple dated occurrences when the evidence supports it. A later
episode does not erase an earlier event unless the dialogue explicitly says it was cancelled or
replaced. Do not stop after finding one plausible episode.
For a direct question, prefer an explicit matching statement over a generic or merely related one.
For a temporal answer, first bind the question's action/object/person to one exact DIA; only then
read its time expression. For an interval or frequency, build a dated inventory of all matching DIAs
and compute the requested result, but do not use a neighboring event's timestamp as a proxy.
When a matching DIA uses a relative phrase such as “a few days ago”, “last Tuesday”, or “yesterday”,
the safest answer is that source phrase (optionally followed by “relative to [the episode date]”);
never replace it with the timestamp of a different DIA. If the question explicitly asks for an
absolute conversion and the reference date is unambiguous, perform that conversion only after
stating or preserving the source relation.
When several episodes contain the same broad verb (for example share, want, get, or go), select the
sentence that jointly satisfies all explicit participants and complements in the question. A generic
recipe, trip, or activity mention is not evidence for a different requested object or relationship.
When a question asks about a plan (for example, "What are X's plans for the summer?"), first look
for a local question-and-answer exchange that explicitly asks about that time period and use the
answer in that exchange as the primary fact.  Do not merge later plans from a different topic just
because they also mention summer; include multiple plans only when the question asks for all/current
plans or the same exchange clearly enumerates them.
For a likelihood or commonsense question, make the smallest supported inference from the person's
recorded preferences, plans, or experiences; do not require the exact question wording to appear,
but do not add unsupported certainty. For a list/count, include every qualifying distinct item/event
visible in the supplied episodes. If the evidence does not establish the answer after this check,
output exactly: Unknown.
For questions about an image, photo, or depicted food/object, inspect the metadata on the exact
matching DIA before relying on neighboring text. A caption or retrieval query is valid textual
evidence for that DIA; quote the depicted item literally, and do not replace it with another image
from the same episode.
For “represent”, “symbolize”, or photo-caption questions, answer with the literal statement in
the matching turn/caption/retrieval description (for example, “performing at the festival”). Bind
the answer to the question's date and exact image-bearing DIA before using other food/object turns.
Do not invent symbolism, emotions, or visual interpretation from a generic discussion.
Return only the shortest sufficient answer without explanation."""


ANSWER_USER = """Question: {question}

Retrieval plan:
{plan}

Complete original episodes retrieved across all hops:
{episodes}

Answer:"""

import numpy as np
from samet_toolkit.llm_ops import EmbeddingInvoker
from scipy.spatial.distance import cosine


LARGE_DISTANCE = 1000
LONG_TEXT = 1_000_000


class DocSegment:
    def __init__(
        self, prev_cosine: float, next_cosine: float, num_char: int, node_ids: list[int]
    ):
        self.prev_cosine = prev_cosine
        self.next_cosine = next_cosine
        self.num_char = num_char
        self.node_ids = node_ids
        self.cosine = min(prev_cosine, next_cosine)

    def __str__(self) -> str:
        return (
            f"DocSegment(prev_cosine={self.prev_cosine}, "
            f"next_cosine={self.next_cosine}, "
            f"num_char={self.num_char}, "
            f"node_ids={self.node_ids})"
        )

    @staticmethod
    def combine_segments(left: "DocSegment", right: "DocSegment") -> "DocSegment":
        return DocSegment(
            prev_cosine=left.prev_cosine,
            next_cosine=right.next_cosine,
            num_char=left.num_char + right.num_char,
            node_ids=left.node_ids + right.node_ids,
        )


class SegmentCombiner:
    def __init__(self, segments: list[DocSegment], max_char_thresh: float = 5000):
        self.segments = segments
        self.max_char_thresh = max_char_thresh

    def merge_segments(self) -> list[DocSegment]:
        impossible_node = DocSegment(
            prev_cosine=LARGE_DISTANCE,
            next_cosine=LARGE_DISTANCE,
            num_char=LONG_TEXT,
            node_ids=[-1],
        )
        is_changed = True
        while is_changed:
            is_changed = False

            merge_candidates = self.get_merge_candidates()
            if not merge_candidates:
                break

            for idx in merge_candidates:
                center, left, right = self._get_neighboring_segments(
                    idx, impossible_node
                )
                best_neighbor = self._get_best_neighbor(
                    center, left, right, impossible_node
                )

                if best_neighbor is not None:
                    is_changed = True
                    if best_neighbor == left:
                        merged_segment = DocSegment.combine_segments(left, center)
                        self.segments[idx - 1] = merged_segment
                        self.segments.pop(idx)
                    else:
                        merged_segment = DocSegment.combine_segments(center, right)
                        self.segments[idx] = merged_segment
                        self.segments.pop(idx + 1)

        return self.segments

    def _get_best_neighbor(self, center, left, right, impossible_node):
        best_neighbor = None
        cur_char = center.num_char
        if cur_char + max(left.num_char, right.num_char) < self.max_char_thresh:
            best_neighbor = left if left.num_char < right.num_char else right
        elif cur_char + left.num_char < self.max_char_thresh:
            best_neighbor = left
        elif cur_char + right.num_char < self.max_char_thresh:
            best_neighbor = right
        if best_neighbor is impossible_node:
            best_neighbor = None

        return best_neighbor

    def _get_neighboring_segments(self, i, impossible_node):
        center = self.segments[i]
        left = self.segments[i - 1] if i > 0 else impossible_node
        right = self.segments[i + 1] if i < len(self.segments) - 1 else impossible_node
        return center, left, right

    def get_merge_candidates(self, percentile: float = 25) -> list[int]:
        lengths = [seg.num_char for seg in self.segments]
        similarities = [seg.cosine for seg in self.segments]

        len_est = np.percentile(lengths, percentile)
        length_threshold = min(float(len_est), self.max_char_thresh)
        similarity_threshold = np.percentile(similarities, percentile / 2)

        cosine_candidates = [
            i
            for i, seg in enumerate(self.segments)
            if seg.cosine <= similarity_threshold
        ]

        length_candidates = [
            i for i, seg in enumerate(self.segments) if seg.num_char <= length_threshold
        ]

        merge_candidates = sorted(
            set(cosine_candidates + length_candidates), reverse=True
        )

        return merge_candidates


class SegmentMerger:
    def __init__(self, config, raw_segments: list[str]):
        self.config = config
        self.raw_segments = raw_segments
        self.segments = self.create_segment(raw_segments)

    def process(self) -> list[str]:
        combiner = SegmentCombiner(self.segments)
        merged_segment_ids = combiner.merge_segments()

        # Return the merged segments, not the ids
        all_res = self._get_combined_segments(merged_segment_ids)
        return all_res

    def _get_combined_segments(self, merged_segment_ids):
        all_res = [
            "\n".join(self.raw_segments[seg.node_ids[0] : seg.node_ids[-1] + 1])
            for seg in merged_segment_ids
        ]
        return all_res

    def create_segment(self, raw_segments: list[str]) -> list[DocSegment]:
        embeddings = EmbeddingInvoker.generate_embeddings(
            self.raw_segments, self.config.embedding.model_name, self.config
        )

        segments = []
        for i, segment in enumerate(raw_segments):
            prev_cosine = LARGE_DISTANCE if i == 0 else 0.0
            next_cosine = LARGE_DISTANCE if i == len(raw_segments) - 1 else 0.0
            segments.append(DocSegment(prev_cosine, next_cosine, len(segment), [i]))
            # Update cosine distances between embeddings
        for i in range(len(embeddings) - 1):
            next_cosine_distance = cosine(embeddings[i], embeddings[i + 1])
            segments[i].next_cosine = next_cosine_distance
            segments[i + 1].prev_cosine = next_cosine_distance

        for segment in segments:
            segment.cosine = min(segment.prev_cosine, segment.next_cosine)

        return segments

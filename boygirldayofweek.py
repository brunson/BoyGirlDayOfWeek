
from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from itertools import chain, product
from random import randint, choice
from textwrap import indent, dedent
from time import time
from typing import NamedTuple, Self, Iterator


def histogram(data: Counter) -> None:
    """
    display a histogram of a Counter object
    """

    count = data.total()
    key_width = max(len(str(k)) for k in data.keys())
    count_width = max(len(str(i)) for i in data.values())
    display_width = 80
    bar_width = display_width - key_width
    num_keys = len(data)
    norm = max((count // num_keys // bar_width, display_width))

    lines = list()
    lines.append(f"{count:,} data points")

    for key in sorted(data.keys()):
        lines.append(f" {str(key):>{key_width}}: [{data[key]:>{count_width}}] {'*' * (data[key]//norm)}")

    output = "\n".join(lines)
    return output


@contextmanager
def timer():
    """
    context manager to time operations and display duration
    """

    start = time()
    print("   == timing... ", end="", flush=True)

    yield

    elapsed = round(time() - start)
    print(f"{elapsed} seconds elapsed ==")


class RandomDate:
    # Beginning and the end of the first quarter of this century
    start = datetime.strptime("Wed Jan  1 00:00:00 2001", "%c")
    end = datetime.strptime("Wed Jan  1 00:00:00 2026", "%c")

    def __init__(self):
        """
        calculate the unix timestamp (seconds since beginning of 1970)
        of start and end arguments

        return a datetime object based on the selection of a random
        second between start and end
        """

        self.dt = datetime.fromtimestamp(
            randint(
                int(self.start.timestamp()),
                int(self.end.timestamp()) - 1,
            )
        )


class Child(NamedTuple):
    gender: str
    birthday: datetime
    rnum: int
    coinflip: str
    days = "sun mon tue wed thu fri sat".split()

    @property
    def birth_day(self):
        return Child.days[self.birthday.dt.weekday()]

    @property
    def birth_month(self):
        return self.birthday.dt.strftime("%m")

    @property
    def birth_year(self):
        return self.birthday.dt.strftime("%Y")


class Siblings:
    genders = "boy girl".split()
    choices = list(range(0, 7))

    def __init__(self):
        self.one = self.make_a_baby()
        self.two = self.make_a_baby()

    def __repr__(self):
        return self.short_label

    def make_a_baby(self):
        return Child(
            choice(self.genders),
            RandomDate(),
            choice(self.choices),
            choice(["heads", "tails"]),
        )

    @property
    def ordered_genders(self) -> tuple[str, str]:
        return (self.one.gender, self.two.gender)

    @property
    def sorted_genders(self) -> tuple[str, str]:
        return tuple(sorted((self.one.gender, self.two.gender)))

    @property
    def random_gender(self) -> str:
        return choice(self.ordered_genders)

    @property
    def label(self):
        return f"{self.one.gender}/{self.two.gender}"

    @property
    def short_label(self):
        return f"{self.one.gender[0]}/{self.two.gender[0]}"


class Dataset:
    def __init__(self, *, data: list[Siblings] = [], size: int = 0):
        if data:
            self.dataset = data
            self.size = len(data)
        else:
            self.generate(size)

    def __repr__(self) -> str:
        return "\n".join(f"{key}: {val}" for (key, val)
                         in self.count_by_label.items())

    def generate(self, size) -> None:
        self.dataset = [Siblings() for _ in range(0, size)]

    def filter(self, filter_func) -> Self:
        filtered = [_ for _ in self.dataset if filter_func(_)]
        return Dataset(data=filtered)

    @property
    def count_by_ordered_genders(self) -> Counter:
        return Counter(sibs.label for sibs in self.dataset)

    @property
    def count_by_genders(self) -> Counter:
        return Counter(_.sorted_genders for _ in self.dataset)

    def iter_individuals(self) -> Iterator[Child]:
        one = (sibs.one for sibs in self.dataset)
        two = (sibs.two for sibs in self.dataset)
        return chain(one, two)

    @property
    def count_by_ind_gender(self) -> Counter:
        return Counter(_.gender for _ in self.iter_individuals())

    @property
    def count_by_birth_day(self) -> Counter:
        return Counter(_.birth_day for _ in self.iter_individuals())

    @property
    def count_by_birth_month(self) -> Counter:
        return Counter(_.birth_day for _ in self.iter_individuals())

    @property
    def count_by_birth_year(self) -> Counter:
        return Counter(_.birth_year for _ in self.iter_individuals())

    @property
    def count_by_label(self) -> Counter:
        return Counter(_.label for _ in self.dataset)

    @property
    def count_by_short_label(self) -> Counter:
        return Counter(_.short_label for _ in self.dataset)


def show_stats(size) -> None:
    """
    Generate random data sets and demonstrate valid distribution
    """

    print(" ==== generating representative date set for statistics ====")
    with timer():
        years = Counter([RandomDate().dt.year for _ in range(0, size)])
    print("\n ==== shows distribution of randomly sampled dates over years ====",
          histogram(years),
          "\n")

    print(" ==== generating sibling set data ====")
    with timer():
        dataset = Dataset(size=size)

    print("\n ==== how are the generated sibling pairs distributed by genders",
          histogram(dataset.count_by_ordered_genders),
          "\n")

    print(" ==== how are the generated offspring distributed by gender",
          histogram(dataset.count_by_ind_gender),
          "\n")

    print(" ==== how are the generated offspring distributed by birth day of week",
          histogram(dataset.count_by_birth_day),
          "\n")

    print(" ==== how are the generated offspring distributed by birth month",
          histogram(dataset.count_by_birth_day),
          "\n")

    print(" ==== how are the generated offspring distributed by birth year",
          histogram(dataset.count_by_birth_year),
          "\n")


def get_dataset(size):
    print(" ==== generating dataset =====")
    with timer():
        dataset = Dataset(size=size)
    print()
    return dataset

@dataclass
class Run:
    """
    base class for a simulation running
    """
    dataset: Dataset
    verbose: bool
    histogram: bool
    counts: Counter = None

    @abstractmethod
    def run(self):
        pass

    def print_header(self):
        name = self.__class__.__name__
        border = "=" * len(name)
        print(f"""\
            {border}
            {name}
            {border}
            """,
            self.__doc__
        )
        print("-\n")

    def print_results(self, filtered, message):
        counts = filtered.count_by_genders
        total = counts.total()
        boy_girl = counts[("boy", "girl")]
        print(message,
              f"{round(boy_girl/total * 100, 2)}%\n boy/girl pairs: {boy_girl}, total: {total}")

        if self.verbose:
            print("\n raw counts:")
            print(indent(str(filtered), "    "))

        if self.histogram:
            print(f"\n{histogram(filtered.count_by_label)}")
            # print(indent(str(filtered), "    "))

        print("\n-\n")


class NoAdditionalInfo(Run):
    """
    The simplest case, shows that the chance of any random set
    of two siblings being different genders is 1/2
    """

    def run(self):
        for known, other in (("boy", "girl"), ("girl", "boy")):
            print(f"I have two children, one is a {known}")
            filtered = self.dataset.filter(lambda _: known in _.ordered_genders)

            self.print_results(
                filtered,
                f"probability my other child is a {other} is"
            )


class OneKnownGender(Run):
    """
    Demonstrate that when we're given the gender of one of them
    siblings we can filter the dataset of any genders that do not
    have at least one sibling of that gender.
    If one is a girl, then two boys is not possible
    The chance of having two different genders goes to 2/3
    """

    def run(self):
        for known, other in (("boy", "girl"), ("girl", "boy")):
            print(f"I have two children, one is a {known}")
            filtered = self.dataset.filter(lambda _: known in _.ordered_genders)

            self.print_results(
                filtered,
                f"probability my other child is a {other} is"
            )


class OneKnownGenderAndDayOfBirth(Run):
    """
    The contentious case: When we use the day of the week of the
    sibling who's gender is known, we filter out all pairs that
    do not include one child of that gender born on that day.

    If there's a girl born on a Tuesday then all pairs that do not
    include a girl born on a Tuesday are removed from our dataset.
    Chance of two different genders DROPS to 14/27 (51.85%)
    """

    def run(self):
        for known, day, other in product(("boy", "girl"), Child.days, ("girl", "boy")):
            print(f"I have two children, one is a {known} born on a {day}")
            filtered = self.dataset.filter(
                lambda _: (
                    (known, day) in ((_.one.gender, _.one.birth_day),
                                     (_.two.gender, _.two.birth_day))
                )
            )
            self.print_results(
                filtered,
                f"probability my other child is a {other} is",
            )


class DayOfBirth(Run):
    """
    Demonstrate that simply filtering by one of the siblings having
    been born on a given day of the week does not change the event
    distribution of the gender pairs. It simply reduces the number
    of pairs in our dataset
    """

    def run(self):
        for day in Child.days:
            print(f"I have two children, one is born on a {day}")
            filtered = self.dataset.filter(
                lambda _: (
                    day in (_.one.birth_day, _.two.birth_day)
                )
            )
            self.print_results(
                filtered,
                "probability my children are different genders is",
            )


class DayOfBirthFirstThenKnownGender(Run):
    """
    Filter first by day of birth, then in a second step apply them
    known gender criteria to show that after those two steps them
    probability of boy/girl siblings is 2/3. Thus, knowing the day
    of birth along with the gender of one child adds no useful
    information
    """

    def run(self):
        for day in Child.days:
            print(f"I have two children, one is born on a {day}, filtering...", end=" ")
            filtered = self.dataset.filter(
                lambda _: (
                    day in (_.one.birth_day, _.two.birth_day)
                )
            )
            print("done")

            print("One of my children is a girl, filtering...", end=" ")
            filtered = filtered.filter(
                lambda _: (
                    "girl" in _.ordered_genders
                )
            )
            print("done")

            self.print_results(
                filtered,
                "probability my other child is a boy is",
            )


class OneKnownGenderAndARandomDay(Run):
    """
    In this simulation we filter based on a given known gender, but
    for each set of siblings we generate a random day of the week
    and select that pair based on it having a child of the given
    gender and the random day of the week we generate for every
    pair.

    This presents the same 14/27 chance as filtering based on day
    of the week born, but doesn't carry the cognitive bias of all
    the known gender children being born on the same day
    """

    def run(self):
        for known, other in product(("boy", "girl"), ("girl", "boy")):
            print(f"I have two children, one is a {known} born on a random day")
            filtered = self.dataset.filter(
                lambda _: (
                    (known, choice(Child.days)) in ((_.one.gender, _.one.birth_day),
                                                    (_.two.gender, _.two.birth_day))
                )
            )
            self.print_results(
                filtered,
                f"probability my other child is a {other} is",
            )


class OneKnownGenderPlusRandomSelection(Run):
    """
    Introduce a simple random number from 0 to 6 as an
    attribute of the child when it's created. Not only is the
    number assigned at birth randomly, but the number that
    determines whether we select the pair changes for every
    set of siblings.

    This case shows that filtering on a random number while
    maintaining the select bias of preserving more of the same
    gender children, almost twice as many, continues to affect
    the calculated probablility of having two different gender
    children even when we're simply using three random values
    between 0 and 6. We obtain the same result: 14/27

    The important factor here is to use the same random number
    to select either or both of the children, because it makes
    it twice as likely to select that pair, thus altering the
    ratio when compared to the total paris in the dataset
    """

    def run(self):
        def random_number(choices=list(range(0, 7))):
            return choice(choices)

        def filter_func(sib):
            rnum = random_number()
            return (known, rnum) in (
                (sib.one.gender, sib.one.rnum),
                (sib.two.gender, sib.two.rnum),
            )

        for known, other in product(("boy", "girl"), ("girl", "boy")):
            print(f"I have two children, one is a {known} and I'm randomly "
                  "selecting one pair in seven")
            filtered = self.dataset.filter(filter_func)
            self.print_results(
                filtered,
                f"probability my other child is a {other} is",
            )


def get_commandline() -> Namespace:
    parser = ArgumentParser("simulator for boy/girl problem")

    parser.add_argument("--stats",
                        action="store_true",
                        help="show distribution of dataset generated")
    parser.add_argument("--dataset-size", "--size", "-s",
                        type=int,
                        default=1_000_000,
                        help="size of random dataset to generate")
    parser.add_argument("--show-raw-counts", "--raw", "-r",
                        action="store_true",
                        help="show distribution of dataset generated")
    parser.add_argument("--histogram", "--hist", "-H",
                        action="store_true",
                        help="show histogram of raw counts")

    return parser.parse_args()


def main():
    args = get_commandline()

    if args.stats:
        show_stats(args.dataset_size)
        raise SystemExit

    dataset = get_dataset(args.dataset_size)
    for run in (
            NoAdditionalInfo,
            OneKnownGender,
            OneKnownGenderAndDayOfBirth,
            DayOfBirth,
            DayOfBirthFirstThenKnownGender,
            OneKnownGenderAndARandomDay,
            OneKnownGenderPlusRandomSelection,
    ):
        action = run(dataset, args.show_raw_counts, args.histogram)
        action.print_header()
        action.run()


if __name__ == "__main__":
    main()

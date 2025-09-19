import pickle

from argparse import ArgumentParser, Namespace
from abc import abstractmethod
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from itertools import chain, product
from pathlib import Path
from random import randint, choice
from textwrap import indent, dedent
from time import time
from typing import Self, Iterator


DAYS: list[str] = "sun mon tue wed thu fri sat".split()


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
        lines.append(f" {str(key):>{key_width}}: [{data[key]:>{count_width}}] "
                     f"{'*' * (data[key]//norm)}")

    output = "\n".join(lines)
    return output


@contextmanager
def timer(message="{} seconds elapsed", wait=False):
    """
    context manager to time operations and display duration
    """

    start = time()
    print("   == timing... ", end="" if wait else "\n", flush=True)

    yield

    elapsed = round(time() - start)
    print(message.format(elapsed))
    print()


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

    def __repr__(self):
        return self.dt.isoformat()


class Child:
    gender: str
    birthdate: datetime
    rnum: int
    cointoss: str

    def __init__(self, gender, birthdate, rnum, cointoss):
        self.gender = gender
        self.birthdate = birthdate
        self.rnum = rnum
        self.cointoss = cointoss
        self.birth_dow = birthdate.dt.strftime("%a").lower()
        self.birthday = birthdate.dt.strftime("%d-%b")
        self.birth_month = birthdate.dt.strftime("%m")
        self.birth_year = birthdate.dt.strftime("%Y")


class Siblings:
    genders = "boy girl".split()
    choices = list(range(0, 7))

    def __init__(self):
        self.one = self.make_a_baby()
        self.two = self.make_a_baby()
        self.ordered_genders = (self.one.gender, self.two.gender)
        self.sorted_genders = tuple(sorted(self.ordered_genders))
        self.label = f"{self.one.gender}/{self.two.gender}"
        self.short_label = f"{self.one.gender[0]}/{self.two.gender[0]}"

    def __repr__(self):
        return f"""{self.one}\n{self.two}"""

    @property
    def random_gender(self) -> str:
        return choice(self.ordered_genders)

    def make_a_baby(self):
        return Child(
            choice(self.genders),
            RandomDate(),
            choice(self.choices),
            choice(["heads", "tails"]),
        )


class Dataset:
    def __init__(self, *, data: list[Siblings] = [], size: int = 0):
        if data:
            self.dataset = data
            self.size = len(data)
        else:
            self.generate(size)

    def __repr__(self) -> str:
        return dedent("\n".join(f"{key:>100}: {val}" for (key, val)
                                in self.count_by_label().items()))

    def generate(self, size) -> None:
        self.dataset = [Siblings() for _ in range(0, size)]

    def filter(self, filter_func) -> Self:
        filtered = [_ for _ in self.dataset if filter_func(_)]
        return Dataset(data=filtered)

    def count_by_label(self) -> Counter:
        return Counter(_.label for _ in self.dataset)

    def count_by_short_label(self) -> Counter:
        return Counter(_.short_label for _ in self.dataset)

    def count_by_ordered_genders(self) -> Counter:
        return Counter(sibs.label for sibs in self.dataset)

    def count_by_genders(self) -> Counter:
        return Counter(_.sorted_genders for _ in self.dataset)

    def iter_individuals(self) -> Iterator[Child]:
        one = (sibs.one for sibs in self.dataset)
        two = (sibs.two for sibs in self.dataset)
        return chain(one, two)

    def count_by_ind_gender(self) -> Counter:
        return Counter(_.gender for _ in self.iter_individuals())

    def count_by_birth_dow(self) -> Counter:
        return Counter(_.birth_dow for _ in self.iter_individuals())

    def count_by_birth_month(self) -> Counter:
        return Counter(_.birth_month for _ in self.iter_individuals())

    def count_by_birth_year(self) -> Counter:
        return Counter(_.birth_year for _ in self.iter_individuals())

    def count_by_birthday(self) -> Counter:
        return Counter(_.birthday for _ in self.iter_individuals())

    def count_by_rnum(self) -> Counter:
        return Counter(_.rnum for _ in self.iter_individuals())

    def count_by_coin_toss(self) -> Counter:
        return Counter(_.cointoss for _ in self.iter_individuals())


def show_stats(dataset: Dataset) -> None:
    """
    Generate random data sets and demonstrate valid distribution
    """

    print("\n ==== shows distribution of randomly sampled dates over years\n",
          histogram(dataset.count_by_birth_year()),
          "\n")

    print("\n ==== how are generated sibling pairs distributed by genders\n",
          histogram(dataset.count_by_ordered_genders()),
          "\n")

    print("\n ==== how are generated offspring distributed by gender\n",
          histogram(dataset.count_by_ind_gender()),
          "\n")

    print(" ==== how are generated offspring distributed by birth year\n",
          histogram(dataset.count_by_birth_year()),
          "\n")

    print(" ==== how are generated offspring distributed by birth month\n",
          histogram(dataset.count_by_birth_month()),
          "\n")

    print(" ==== how are generated offspring distributed by day of week\n",
          histogram(dataset.count_by_birth_dow()),
          "\n")

    print(" ==== how are generated offspring distributed by birthday\n",
          histogram(dataset.count_by_birthday()),
          "\n")

    print(" ==== how are generated offspring distributed by rand number IaB\n",
          histogram(dataset.count_by_rnum()),
          "\n")

    print(" ==== how are generated offspring distributed by coin toss IaB\n",
          histogram(dataset.count_by_coin_toss()),
          "\n")


def get_dataset(
        size: int,
        regen: bool = False,
        cache: bool = True
) -> Dataset:
    cache = Path("~/.boygirl.cached").expanduser()

    if cache.exists() and not regen:
        print(" ==== reading cached dataset =====")
        with timer("cached dataset loaded in {} seconds"):
            dataset = pickle.load(cache.open("rb"))
    else:
        print(" ==== generating dataset =====")
        with timer("dataset generated in {} seconds"):
            dataset = Dataset(size=size)

        if cache:
            with timer("dataset saved in {} seconds"):
                pickle.dump(dataset, cache.open("wb"))

    print()
    return dataset


@dataclass
class Simulation:
    """
    base class for a simulation running
    """
    dataset: Dataset
    args: Namespace
    formula: str = ""
    expected: float = 0.0
    counts: Counter = None
    percentages: list[float] = None

    def run(self):
        self.percentages = list()
        self.print_header()
        self.run_simulation()
        self.print_footer()

    @abstractmethod
    def run_simulation(self):
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
        counts = filtered.count_by_genders()
        total = counts.total()
        boy_girl = counts[("boy", "girl")]
        self.percentages.append(boy_girl/total)
        print(message,
              f"{round(boy_girl/total * 100, 2)}%\n boy/girl pairs: "
              f"{boy_girl}, total: {total}")

        if self.args.raw_counts or self.args.verbose:
            print("\n raw counts")
            print(indent(str(filtered), "  "))

        if self.args.histogram or self.args.verbose:
            print(f"\n{histogram(filtered.count_by_label())}")

        print("\n-\n")

    def print_footer(self):
        average = round(sum(self.percentages) / len(self.percentages) * 100, 2)
        print(f"average of all the probabilities is {average}%")
        print(f"the expected value is {self.formula} or "
              f"{self.expected}%\n\n----\n")


class NoAdditionalInfo(Simulation):
    """
    The simplest case, shows that the chance of any random set of
    two siblings being different genders is 1/2

    """

    def __post_init__(self):
        self.formula = "1/2"
        self.expected = 1/2

    def run_simulation(self):
        print("I have two children")
        filtered = self.dataset

        self.print_results(
            filtered,
            "probability I have a boy and a girl is"
        )


class OneKnownGender(Simulation):
    """
    Demonstrate that when we're given the gender of one of them
    siblings we can filter the dataset of any genders that do not have
    at least one sibling of that gender.  If one is a girl, then two
    boys is not possible The chance of having two different genders
    goes to 2/3

    """

    def run_simulation(self):
        for known, other in (("boy", "girl"), ("girl", "boy")):
            print(f"I have two children, one is a {known}")
            filtered = self.dataset.filter(
                lambda _: known in _.ordered_genders
            )

            self.print_results(
                filtered,
                f"probability my other child is a {other} is"
            )


class OneKnownGenderAndDayOfBirth(Simulation):
    """
    The contentious case:

    When we use the gender combined with the day of the week of the
    day of the week they were born, we filter out all pairs that do
    not include one child of that gender born on that day.

    If there's a girl born on a Tuesday then all pairs that do not
    include a girl born on a Tuesday are removed from our dataset.
    Chance of two different genders DROPS to 14/27 (51.85%)

    """

    def run_simulation(self):
        for known, day, other in product(("boy", "girl"),
                                         DAYS,
                                         ("girl", "boy")):
            print(f"I have two children, one is a {known} born on a {day}")
            filtered = self.dataset.filter(
                lambda _: (
                    (known, day) in ((_.one.gender, _.one.birth_dow),
                                     (_.two.gender, _.two.birth_dow))
                )
            )
            self.print_results(
                filtered,
                f"probability my other child is a {other} is",
            )


class DayOfBirth(Simulation):
    """
    Demonstrate that simply filtering by one of the siblings
    having been born on a given day of the week does not change the
    event distribution of the gender pairs. It simply reduces the
    number of pairs in our dataset

    """

    def run_simulation(self):
        for day in DAYS:
            print(f"I have two children, one is born on a {day}")
            filtered = self.dataset.filter(
                lambda _: (
                    day in (_.one.birth_dow, _.two.birth_dow)
                )
            )
            self.print_results(
                filtered,
                "probability my children are different genders is",
            )


class DayOfBirthFirstThenKnownGender(Simulation):
    """
    Filter first by day of birth, then in a second step apply them
    known gender criteria to show that after those two steps them
    probability of boy/girl siblings is 2/3. Thus, knowing the day of
    birth along with the gender of one child adds no useful
    information

    """

    def run_simulation(self):
        for day in DAYS:
            print(f"I have two children, one is born on a {day}, filtering...",
                  end=" ")
            filtered = self.dataset.filter(
                lambda _: (
                    day in (_.one.birth_dow, _.two.birth_dow)
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


class OneKnownGenderAndARandomDay(Simulation):
    """
    In this simulation we filter based on a given known gender,
    but for each set of siblings we generate a random day of the week
    and select that pair based on it having a child of the given
    gender and the random day of the week we generate for every pair.

    This presents the same 14/27 chance as filtering based on day
    of the week born, but doesn't carry the cognitive bias of all
    the known gender children being born on the same day

    """

    def run_simulation(self):
        for known, other in product(("boy", "girl"), ("girl", "boy")):
            print(f"I have two children, one is a {known} "
                  "born on a random day")
            filtered = self.dataset.filter(
                lambda _: (
                    (known, choice(DAYS)) in ((_.one.gender, _.one.birth_dow),
                                              (_.two.gender, _.two.birth_dow))
                )
            )
            self.print_results(
                filtered,
                f"probability my other child is a {other} is",
            )


class OneKnownGenderPlusRandomSelection(Simulation):
    """
    Introduce a simple random number from 0 to 6 as an attribute
    of the child when it's created. Not only is the number assigned at
    birth randomly, but the number that determines whether we select
    the pair changes for every set of siblings.

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

    def run_simulation(self):
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


class OneKnownGenderPlusCoinToss(Simulation):
    """
    Introduce a coin toss as attribute of the child when it's
    created. I tossed a coin when each of my kids was born and
    tattooed it on their forehead

    """

    def run_simulation(self):
        for known, coin, other in product(("boy", "girl"),
                                          ("heads", "tails"),
                                          ("girl", "boy")):
            print(f"I have two children, one is a {known} and their "
                  f"coin toss was {coin}")
            filtered = self.dataset.filter(
                lambda _, known=known, coin=coin:
                (known, coin) in (
                    (_.one.gender, _.one.cointoss),
                    (_.two.gender, _.two.cointoss),
                )
            )
            self.print_results(
                filtered,
                f"probability my other child is a {other} is",
            )


class OneKnownGenderPlusRandomCoinToss(Simulation):
    """
    Same coin toss, but a random coin toss is generated every time
    we select a sibling pair. So, for each pair of siblings toss a
    coin and keep the pair if either child is of the opposite gender
    and matches the coin that was just tossed.

    """

    def run_simulation(self):
        def flip_a_coin(choices="heads tails".split()):
            return choice(choices)

        def filter_func(sib):
            toss = flip_a_coin()
            return (known, toss) in (
                (sib.one.gender, sib.one.cointoss),
                (sib.two.gender, sib.two.cointoss),
            )

        for known, other in product(("boy", "girl"), ("girl", "boy")):
            print(f"I have two children, one is a {known} and "
                  "you're flipping a coin for every set of kids\n"
                  "to decide if they're part of our dataset")
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
                        default=100_000,
                        help="size of random dataset to generate")
    parser.add_argument("--raw-counts", "--raw", "-r",
                        action="store_true",
                        help="show distribution of dataset generated")
    parser.add_argument("--histogram", "--hist", "-H",
                        action="store_true",
                        help="show histogram of raw counts")
    parser.add_argument("--verbose", "-v",
                        action="store_true",
                        help="generate all output")
    parser.add_argument("--regenerate-cache", "--regen", "-R",
                        action="store_true",
                        help="regenerate cache")
    parser.add_argument("--show-timings", "--timer", "-T",
                        action="store_true",
                        help="regenerate cache")

    return parser.parse_args()


def main():
    args = get_commandline()

    dataset = get_dataset(args.dataset_size, args.regenerate_cache)

    if args.stats or args.verbose:
        show_stats(dataset)
        if args.stats:
            raise SystemExit

    for simulation in (
            NoAdditionalInfo,
            OneKnownGender,
            OneKnownGenderAndDayOfBirth,
            DayOfBirth,
            DayOfBirthFirstThenKnownGender,
            OneKnownGenderAndARandomDay,
            OneKnownGenderPlusRandomSelection,
            OneKnownGenderPlusCoinToss,
            OneKnownGenderPlusRandomCoinToss,
    ):
        if args.show_timings:
            with timer():
                simulation(dataset, args).run()
        else:
            simulation(dataset, args).run()


if __name__ == "__main__":
    main()

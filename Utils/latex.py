from amplify import Poly, ConstraintList


def is_number(s: str):
    try:
        float(s)
        return True
    except ValueError:
        return False


def obj_to_latex(obj: Poly, short=False):
    """
    Convert Objective Function to LaTeX, with terms and variable count

    Args:
        obj (Poly): Objective Function
        short (bool, optional): If set to True, will print only terms and variable count. Defaults to False.

    Returns:
        str: String in Markdown LaTeX format
    """

    terms = str(obj).split("+")

    variable = set()

    def join_cdot(term):
        t = term.strip().split(" ")

        vars = list(filter(lambda x: not is_number(x),
                           map(lambda x: x.strip(), t)))
        for var in vars:
            variable.add(var)

        return " \\cdot ".join(t)

    terms_cdot = list(map(join_cdot, terms))

    long_part = "$" + " + ".join(terms_cdot) + "$"

    return f"Terms Count = {len(terms_cdot)}, Unique Variable = {len(variable)}  \n" + (long_part if not short else "")


def con_to_latex(con: ConstraintList, short=False):
    """Convert constraint list to LaTeX

    Args:
        con (ConstraintList): Constraint List
        short (bool, optional): If set to True, will print only constraint count. Defaults to False.

    Returns:
        str: String in Markdown LaTeX format
    """

    con_count = f"Constraint Count: {len(con)}  \n"

    if short:
        return con_count

    md = ''
    for c in con:
        lhs, rhs = str(c).split("(")
        lhs = lhs.replace("<=", "\\le").replace(
            ">=", "\\ge").replace("==", "=")
        md += f"${lhs.strip()}$ ({rhs}  \n"

    return con_count + md
